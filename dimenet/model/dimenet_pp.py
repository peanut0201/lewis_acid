import tensorflow as tf

from .layers.embedding_block import EmbeddingBlock
from .layers.bessel_basis_layer import BesselBasisLayer
from .layers.spherical_basis_layer import SphericalBasisLayer
from .layers.interaction_pp_block import InteractionPPBlock
from .layers.output_pp_block import OutputPPBlock
from .activations import swish


class DimeNetPP(tf.keras.Model):
    """
    DimeNet++ model.

    Parameters
    ----------
    emb_size
        Embedding size used for the messages
    out_emb_size
        Embedding size used for atoms in the output block
    int_emb_size
        Embedding size used for interaction triplets
    basis_emb_size
        Embedding size used inside the basis transformation
    num_blocks
        Number of building blocks to be stacked
    num_spherical
        Number of spherical harmonics
    num_radial
        Number of radial basis functions
    envelope_exponent
        Shape of the smooth cutoff
    cutoff
        Cutoff distance for interatomic interactions
    num_before_skip
        Number of residual layers in interaction block before skip connection
    num_after_skip
        Number of residual layers in interaction block after skip connection
    num_dense_output
        Number of dense layers for the output blocks
    num_targets
        Number of targets to predict
    activation
        Activation function
    extensive
        Whether the output should be extensive (proportional to the number of atoms)
    output_init
        Initialization method for the output layer (last layer in output block)
    """

    def __init__(
            self, emb_size, out_emb_size, int_emb_size, basis_emb_size,
            num_blocks, num_spherical, num_radial,
            cutoff=5.0, envelope_exponent=5, num_before_skip=1,
            num_after_skip=2, num_dense_output=3, num_targets=None,
            activation=swish, extensive=True, output_init='zeros',
            freeze_backbone=False, name='dimenet', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_blocks = num_blocks
        self.extensive = extensive
        self.freeze_backbone = freeze_backbone
        
        # Set num_targets with default value if not provided
        self.num_targets = num_targets if num_targets is not None else 4

        # Cosine basis function expansion layer
        self.rbf_layer = BesselBasisLayer(
            num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)
        self.sbf_layer = SphericalBasisLayer(
            num_spherical, num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)

        # Embedding and first output block
        self.output_blocks = []
        self.emb_block = EmbeddingBlock(emb_size, activation=activation)
        self.output_blocks.append(
            OutputPPBlock(emb_size, out_emb_size, num_dense_output, self.num_targets,
                          activation=activation, output_init=output_init))

        # Interaction and remaining output blocks
        self.int_blocks = []
        for i in range(num_blocks):
            self.int_blocks.append(
                InteractionPPBlock(emb_size, int_emb_size, basis_emb_size, num_before_skip,
                                   num_after_skip, activation=activation))
            self.output_blocks.append(
                OutputPPBlock(emb_size, out_emb_size, num_dense_output, self.num_targets,
                              activation=activation, output_init=output_init))

    def set_num_targets(self, num_targets):
        """Dynamically set the number of targets and rebuild output blocks"""
        self.num_targets = num_targets
        # Note: This method sets the target count but doesn't rebuild the model
        # The model should be recreated with the new num_targets for full functionality
        print(f"目标数量已设置为: {self.num_targets}")

    def calculate_interatomic_distances(self, R, idx_i, idx_j):
        Ri = tf.gather(R, idx_i)
        Rj = tf.gather(R, idx_j)
        # ReLU prevents negative numbers in sqrt
        Dij = tf.sqrt(tf.nn.relu(tf.reduce_sum((Ri - Rj)**2, -1)))
        return Dij

    def calculate_neighbor_angles(self, R, id3_i, id3_j, id3_k):
        """Calculate angles for neighboring atom triplets"""
        Ri = tf.gather(R, id3_i)
        Rj = tf.gather(R, id3_j)
        Rk = tf.gather(R, id3_k)
        R1 = Rj - Ri
        R2 = Rk - Rj
        x = tf.reduce_sum(R1 * R2, axis=-1)
        y = tf.linalg.cross(R1, R2)
        y = tf.norm(y, axis=-1)
        angle = tf.math.atan2(y, x)
        return angle

    def call(self, inputs):
        Z, R                         = inputs['Z'], inputs['R']
        batch_seg                    = inputs['batch_seg']
        idnb_i, idnb_j               = inputs['idnb_i'], inputs['idnb_j']
        id_expand_kj, id_reduce_ji   = inputs['id_expand_kj'], inputs['id_reduce_ji']
        id3dnb_i, id3dnb_j, id3dnb_k = inputs['id3dnb_i'], inputs['id3dnb_j'], inputs['id3dnb_k']
        n_atoms = tf.shape(Z)[0]

        # Calculate distances
        Dij = self.calculate_interatomic_distances(R, idnb_i, idnb_j)
        rbf = self.rbf_layer(Dij)

        # Calculate angles
        Anglesijk = self.calculate_neighbor_angles(
            R, id3dnb_i, id3dnb_j, id3dnb_k)
        sbf = self.sbf_layer([Dij, Anglesijk, id_expand_kj])

        # Embedding block
        x = self.emb_block([Z, rbf, idnb_i, idnb_j])
        P = self.output_blocks[0]([x, rbf, idnb_i, n_atoms])

        # Interaction blocks
        for i in range(self.num_blocks):
            x = self.int_blocks[i]([x, rbf, sbf, id_expand_kj, id_reduce_ji])
            P += self.output_blocks[i+1]([x, rbf, idnb_i, n_atoms])

        if self.extensive:
            P = tf.math.segment_sum(P, batch_seg)
        else:
            P = tf.math.segment_mean(P, batch_seg)
        return P

    def freeze_backbone_layers(self):
        """冻结主干网络层(embedding和interaction blocks)，保留输出层可训练"""
        # 冻结embedding block
        self.emb_block.trainable = False
        
        # 冻结interaction blocks
        for int_block in self.int_blocks:
            int_block.trainable = False
            
        # 冻结basis layers
        self.rbf_layer.trainable = False
        self.sbf_layer.trainable = False
        
        # 保持输出层可训练
        for output_block in self.output_blocks:
            output_block.trainable = True
        
        print("主干网络已冻结，只训练输出头")
        
    def unfreeze_backbone_layers(self):
        """解冻主干网络层"""
        # 解冻embedding block
        self.emb_block.trainable = True
        
        # 解冻interaction blocks
        for int_block in self.int_blocks:
            int_block.trainable = True
            
        # 解冻basis layers
        self.rbf_layer.trainable = True
        self.sbf_layer.trainable = True
        
        print("主干网络已解冻")
        
    def get_trainable_params_count(self):
        """获取可训练参数数量"""
        trainable_params = 0
        
        # 检查所有子模块
        modules_to_check = [
            self.emb_block,
            self.rbf_layer,
            self.sbf_layer,
            *self.int_blocks,
            *self.output_blocks
        ]
        
        for module in modules_to_check:
            if hasattr(module, 'trainable_variables') and module.trainable:
                try:
                    for var in module.trainable_variables:
                        trainable_params += tf.size(var).numpy()
                except Exception as e:
                    # 如果trainable_variables还没有创建，跳过
                    pass
        
        return trainable_params
    
    def print_trainable_layers(self):
        """打印可训练层的信息"""
        print("=== 可训练层信息 ===")
        
        modules_to_check = [
            ("emb_block", self.emb_block),
            ("rbf_layer", self.rbf_layer),
            ("sbf_layer", self.sbf_layer),
        ]
        
        # 添加interaction blocks
        for i, int_block in enumerate(self.int_blocks):
            modules_to_check.append((f"int_block_{i}", int_block))
        
        # 添加output blocks
        for i, output_block in enumerate(self.output_blocks):
            modules_to_check.append((f"output_block_{i}", output_block))
        
        for name, module in modules_to_check:
            if hasattr(module, 'trainable_variables'):
                if module.trainable:
                    param_count = sum(tf.size(var).numpy() for var in module.trainable_variables)
                    print(f"  {name}: 可训练, 参数数量: {param_count}")
                else:
                    print(f"  {name}: 已冻结")
            else:
                print(f"  {name}: 无trainable_variables属性")


def create_dimenet_pp_from_data_container(data_container, freeze_backbone=False, **model_kwargs):
    """
    根据数据容器动态创建DimeNet++模型
    
    Parameters:
    data_container : DataContainer
        包含目标键信息的数据容器
    freeze_backbone : bool
        是否冻结主干网络，只训练输出头
    **model_kwargs
        传递给DimeNetPP的其他参数
        
    Returns:
    DimeNetPP
        配置了正确目标数量的模型
    """
    num_targets = len(data_container.target_keys)
    print(f"根据数据容器创建模型，目标数量: {num_targets}")
    print(f"目标键: {data_container.target_keys}")
    
    model = DimeNetPP(num_targets=num_targets, freeze_backbone=freeze_backbone, **model_kwargs)
    
    # 如果设置了冻结主干，则冻结相关层
    if freeze_backbone:
        model.freeze_backbone_layers()
        print("主干网络已冻结，只训练输出头")
    
    return model

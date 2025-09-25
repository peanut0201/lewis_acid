import tensorflow as tf
from .schedules import LinearWarmupExponentialDecay


class Trainer:
    def __init__(self, model, learning_rate=1e-3, warmup_steps=None,
                 decay_steps=100000, decay_rate=0.96,
                 ema_decay=0.999, max_grad_norm=10.0, 
                 loss_type='mae', target_weights=None, freeze_backbone=False):
        self.model = model
        self.ema_decay = ema_decay
        self.max_grad_norm = max_grad_norm
        self.loss_type = loss_type
        self.target_weights = target_weights
        self.freeze_backbone = freeze_backbone

        if warmup_steps is not None:
            self.learning_rate = LinearWarmupExponentialDecay(
                learning_rate, warmup_steps, decay_steps, decay_rate)
        else:
            self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps, decay_rate)

        # 使用legacy优化器以提高M1/M2 Mac性能
        # 如果冻结主干，使用更小的学习率
        if freeze_backbone:
            # 对于学习率调度器，需要创建新的调度器
            if warmup_steps is not None:
                actual_learning_rate = LinearWarmupExponentialDecay(
                    learning_rate * 0.1, warmup_steps, decay_steps, decay_rate)
            else:
                actual_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                    learning_rate * 0.1, decay_steps, decay_rate)
        else:
            actual_learning_rate = self.learning_rate
            
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=actual_learning_rate, amsgrad=True)
        
        # 手动实现EMA (Exponential Moving Average)
        self.ema_decay = ema_decay
        self.ema_vars = None

        # Initialize backup variables
        if model.built:
            self.backup_vars = [tf.Variable(var, dtype=var.dtype, trainable=False)
                                for var in self.model.trainable_weights]
        else:
            self.backup_vars = None

    def compute_loss(self, targets, preds):
        """计算损失函数"""
        if self.loss_type == 'mae':
            # 平均绝对误差
            mae = tf.reduce_mean(tf.abs(targets - preds), axis=0)
            if self.target_weights is not None:
                # 加权MAE
                mae = mae * self.target_weights
            mean_mae = tf.reduce_mean(mae)
            return mean_mae, mae
        elif self.loss_type == 'mse':
            # 均方误差
            mse = tf.reduce_mean(tf.square(targets - preds), axis=0)
            if self.target_weights is not None:
                mse = mse * self.target_weights
            mean_mse = tf.reduce_mean(mse)
            return mean_mse, tf.sqrt(mse)  # 返回RMSE作为MAE
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def update_weights(self, loss, gradient_tape):
        # 获取可训练的参数
        trainable_vars = self.model.trainable_weights
        
        # 如果冻结主干，只计算输出头的梯度
        if self.freeze_backbone:
            # 只对输出块计算梯度
            output_vars = []
            for layer in self.model.layers:
                if hasattr(layer, 'name') and 'output' in layer.name.lower():
                    output_vars.extend(layer.trainable_variables)
            
            if output_vars:
                grads = gradient_tape.gradient(loss, output_vars)
                # 创建完整的梯度列表，冻结的参数梯度为None
                full_grads = []
                for var in trainable_vars:
                    if var in output_vars:
                        idx = output_vars.index(var)
                        full_grads.append(grads[idx])
                    else:
                        full_grads.append(None)
            else:
                # 如果没有找到输出层，使用所有可训练参数
                grads = gradient_tape.gradient(loss, trainable_vars)
                full_grads = grads
        else:
            # 正常训练所有参数
            grads = gradient_tape.gradient(loss, trainable_vars)
            full_grads = grads

        # 过滤掉None梯度
        filtered_grads = [g for g in full_grads if g is not None]
        filtered_vars = [v for g, v in zip(full_grads, trainable_vars) if g is not None]
        
        if filtered_grads:
            global_norm = tf.linalg.global_norm(filtered_grads)
            if self.max_grad_norm is not None:
                filtered_grads, _ = tf.clip_by_global_norm(filtered_grads, self.max_grad_norm, use_norm=global_norm)

            self.optimizer.apply_gradients(zip(filtered_grads, filtered_vars))
        
        # 更新EMA变量
        self._update_ema_vars()

    def _update_ema_vars(self):
        """更新EMA变量"""
        if self.ema_vars is None:
            # 初始化EMA变量
            self.ema_vars = [tf.Variable(var, dtype=var.dtype, trainable=False)
                           for var in self.model.trainable_weights]
        else:
            # 更新EMA变量
            for var, ema_var in zip(self.model.trainable_weights, self.ema_vars):
                ema_var.assign(self.ema_decay * ema_var + (1 - self.ema_decay) * var)

    def load_averaged_variables(self):
        """加载EMA变量到模型参数"""
        if self.ema_vars is not None:
            for var, ema_var in zip(self.model.trainable_weights, self.ema_vars):
                var.assign(ema_var)

    def save_variable_backups(self):
        if self.backup_vars is None:
            self.backup_vars = [tf.Variable(var, dtype=var.dtype, trainable=False)
                                for var in self.model.trainable_weights]
        else:
            for var, bck in zip(self.model.trainable_weights, self.backup_vars):
                bck.assign(var)

    def restore_variable_backups(self):
        for var, bck in zip(self.model.trainable_weights, self.backup_vars):
            var.assign(bck)

    @tf.function
    def train_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        with tf.GradientTape() as tape:
            preds = self.model(inputs, training=True)
            # 使用新的损失计算方法
            loss, mae = self.compute_loss(targets, preds)
        self.update_weights(loss, tape)

        nsamples = tf.shape(preds)[0]
        metrics.update_state(loss, loss, mae, nsamples)

        return loss

    @tf.function
    def test_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        preds = self.model(inputs, training=False)
        # 使用新的损失计算方法
        loss, mae = self.compute_loss(targets, preds)

        nsamples = tf.shape(preds)[0]
        metrics.update_state(loss, loss, mae, nsamples)

        return loss

    @tf.function
    def predict_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        preds = self.model(inputs, training=False)

        # 使用新的损失计算方法
        loss, mae = self.compute_loss(targets, preds)
        nsamples = tf.shape(preds)[0]
        metrics.update_state(loss, loss, mae, nsamples)

        return preds

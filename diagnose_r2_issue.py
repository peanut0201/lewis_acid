#!/usr/bin/env python3
"""
诊断R²为负值的问题
"""

import tensorflow as tf
import numpy as np
import yaml
import ast
import os
from sklearn.metrics import r2_score, mean_absolute_error

from dimenet.model.dimenet_pp import create_dimenet_pp_from_data_container
from dimenet.model.activations import swish
from dimenet.training.trainer import Trainer
from dimenet.training.metrics import Metrics
from dimenet.training.data_container import DataContainer
from dimenet.training.data_provider import DataProvider

def main():
    print("=== R²问题诊断 ===")
    
    # 加载配置
    with open('config_pp.yaml', 'r') as c:
        config = yaml.safe_load(c)
    
    for key, val in config.items():
        if type(val) is str:
            try:
                config[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                pass
    
    # 创建数据容器
    data_container = DataContainer(
        json_file='data/FIA49k_Al.json', 
        xlsx_file='data/FIA49k.xlsx', 
        cutoff=config['cutoff'], 
        target_keys=None
    )
    
    print(f"✅ 数据容器创建完成")
    print(f"   目标键: {data_container.target_keys}")
    print(f"   数据量: {len(data_container)}")
    
    # 创建模型
    model = create_dimenet_pp_from_data_container(
        data_container,
        emb_size=config['emb_size'], 
        out_emb_size=config['out_emb_size'],
        int_emb_size=config['int_emb_size'], 
        basis_emb_size=config['basis_emb_size'],
        num_blocks=config['num_blocks'], 
        num_spherical=config['num_spherical'], 
        num_radial=config['num_radial'],
        cutoff=config['cutoff'], 
        envelope_exponent=config['envelope_exponent'],
        num_before_skip=config['num_before_skip'], 
        num_after_skip=config['num_after_skip'],
        num_dense_output=config['num_dense_output'],
        activation=swish, 
        extensive=config['extensive'], 
        output_init=config['output_init'],
        freeze_backbone=config['freeze_backbone']
    )
    
    # 构建模型
    data_provider = DataProvider(
        data_container, 
        config['num_train'], 
        config['num_valid'], 
        config['batch_size'],
        seed=config['data_seed'], 
        randomized=True
    )
    train_dataset = data_provider.get_dataset('train').prefetch(tf.data.experimental.AUTOTUNE)
    train_dataset_iter = iter(train_dataset)
    inputs, targets = next(train_dataset_iter)
    _ = model(inputs)  # 构建模型
    
    print("✅ 模型构建完成")
    
    # 创建训练器
    trainer = Trainer(
        model, 
        config['learning_rate'], 
        config['warmup_steps'],
        config['decay_steps'], 
        config['decay_rate'],
        ema_decay=config['ema_decay'], 
        max_grad_norm=1000,
        freeze_backbone=config['freeze_backbone']
    )
    
    # 加载最佳模型权重
    best_ckpt_file = 'logs_20250922_234938/best_ckpt'
    if os.path.exists(best_ckpt_file + '.index'):
        print(f"📂 加载最佳模型权重: {best_ckpt_file}")
        model.load_weights(best_ckpt_file)
        print("✅ 最佳模型权重加载完成")
    else:
        print("❌ 未找到最佳模型权重文件")
        return
    
    # 在测试集上评估
    print("\n=== 测试集评估 ===")
    test_dataset = data_provider.get_dataset('test').prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset_iter = iter(test_dataset)
    
    all_predictions = []
    all_targets = []
    
    num_test = data_provider.nsamples['test']
    num_batches = int(np.ceil(num_test / config['batch_size']))
    
    print(f"测试集样本数: {num_test}")
    print(f"评估批次数: {num_batches}")
    
    for i in range(num_batches):
        try:
            inputs, targets = next(test_dataset_iter)
            preds = model(inputs, training=False)
            
            all_predictions.append(preds.numpy())
            all_targets.append(targets.numpy())
            
            if (i + 1) % 5 == 0:
                print(f"  已处理 {i + 1}/{num_batches} 批次")
                
        except StopIteration:
            break
        except Exception as e:
            print(f"测试批次 {i} 出错: {e}")
            break
    
    if all_predictions and all_targets:
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        print(f"\n=== 预测结果分析 ===")
        print(f"预测值形状: {all_predictions.shape}")
        print(f"真实值形状: {all_targets.shape}")
        
        print(f"\n预测值统计:")
        print(f"  范围: [{all_predictions.min():.4f}, {all_predictions.max():.4f}]")
        print(f"  均值: {all_predictions.mean(axis=0)}")
        print(f"  标准差: {all_predictions.std(axis=0)}")
        
        print(f"\n真实值统计:")
        print(f"  范围: [{all_targets.min():.4f}, {all_targets.max():.4f}]")
        print(f"  均值: {all_targets.mean(axis=0)}")
        print(f"  标准差: {all_targets.std(axis=0)}")
        
        print(f"\n=== 各目标详细分析 ===")
        target_keys = data_container.target_keys
        
        for i, target_key in enumerate(target_keys):
            pred_i = all_predictions[:, i]
            true_i = all_targets[:, i]
            
            # 计算各种指标
            r2 = r2_score(true_i, pred_i)
            mae = mean_absolute_error(true_i, pred_i)
            mse = np.mean((pred_i - true_i) ** 2)
            
            # 计算基准线（均值预测）
            mean_pred = np.full_like(true_i, true_i.mean())
            r2_baseline = r2_score(true_i, mean_pred)
            mae_baseline = mean_absolute_error(true_i, mean_pred)
            
            print(f"\n{target_key}:")
            print(f"  R²: {r2:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  MSE: {mse:.4f}")
            print(f"  预测值范围: [{pred_i.min():.4f}, {pred_i.max():.4f}]")
            print(f"  真实值范围: [{true_i.min():.4f}, {true_i.max():.4f}]")
            print(f"  预测值均值: {pred_i.mean():.4f}")
            print(f"  真实值均值: {true_i.mean():.4f}")
            print(f"  基准线R² (均值预测): {r2_baseline:.4f}")
            print(f"  基准线MAE (均值预测): {mae_baseline:.4f}")
            
            # 分析问题
            if r2 < 0:
                print(f"  ❌ R²为负值！模型预测比均值预测还差")
                print(f"  💡 可能原因:")
                print(f"     - 模型预测值范围异常")
                print(f"     - 预测值与真实值相关性极差")
                print(f"     - 模型权重加载有问题")
            else:
                print(f"  ✅ R²正常")
        
        # 整体分析
        print(f"\n=== 整体分析 ===")
        overall_r2 = r2_score(all_targets.flatten(), all_predictions.flatten())
        overall_mae = mean_absolute_error(all_targets.flatten(), all_predictions.flatten())
        
        print(f"整体R²: {overall_r2:.4f}")
        print(f"整体MAE: {overall_mae:.4f}")
        
        if overall_r2 < 0:
            print(f"\n❌ 整体R²为负值！")
            print(f"💡 可能的问题:")
            print(f"   1. 模型权重加载错误")
            print(f"   2. 数据预处理问题")
            print(f"   3. 模型架构不匹配")
            print(f"   4. 训练过程中出现严重问题")
            
            # 检查预测值是否异常
            if np.allclose(all_predictions, 0):
                print(f"   🔍 预测值全为0，可能是权重加载问题")
            elif np.std(all_predictions) < 0.01:
                print(f"   🔍 预测值方差极小，可能是模型退化")
            elif np.abs(all_predictions.mean()) > 10:
                print(f"   🔍 预测值均值异常大，可能是数值不稳定")
        else:
            print(f"✅ 整体R²正常")

if __name__ == "__main__":
    main()

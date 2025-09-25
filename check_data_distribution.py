#!/usr/bin/env python3
"""
检查验证集和测试集的数据分布差异
"""

import tensorflow as tf
import numpy as np
import yaml
import ast

from dimenet.training.data_container import DataContainer
from dimenet.training.data_provider import DataProvider

def main():
    print("=== 数据分布差异检查 ===")
    
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
    
    # 创建数据提供器
    data_provider = DataProvider(
        data_container, 
        config['num_train'], 
        config['num_valid'], 
        config['batch_size'],
        seed=config['data_seed'], 
        randomized=True
    )
    
    print(f"数据集划分:")
    print(f"  训练集: {data_provider.nsamples['train']}")
    print(f"  验证集: {data_provider.nsamples['val']}")
    print(f"  测试集: {data_provider.nsamples['test']}")
    
    # 收集验证集数据
    print("\n=== 收集验证集数据 ===")
    val_dataset = data_provider.get_dataset('val').prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset_iter = iter(val_dataset)
    
    val_targets = []
    num_val_batches = int(np.ceil(data_provider.nsamples['val'] / config['batch_size']))
    
    for i in range(num_val_batches):
        try:
            inputs, targets = next(val_dataset_iter)
            val_targets.append(targets.numpy())
        except StopIteration:
            break
    
    val_targets = np.concatenate(val_targets, axis=0)
    
    # 收集测试集数据
    print("=== 收集测试集数据 ===")
    test_dataset = data_provider.get_dataset('test').prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset_iter = iter(test_dataset)
    
    test_targets = []
    num_test_batches = int(np.ceil(data_provider.nsamples['test'] / config['batch_size']))
    
    for i in range(num_test_batches):
        try:
            inputs, targets = next(test_dataset_iter)
            test_targets.append(targets.numpy())
        except StopIteration:
            break
    
    test_targets = np.concatenate(test_targets, axis=0)
    
    print(f"\n=== 数据分布对比 ===")
    target_keys = data_container.target_keys
    
    for i, target_key in enumerate(target_keys):
        val_data = val_targets[:, i]
        test_data = test_targets[:, i]
        
        print(f"\n{target_key}:")
        print(f"  验证集: 范围=[{val_data.min():.4f}, {val_data.max():.4f}], 均值={val_data.mean():.4f}, 标准差={val_data.std():.4f}")
        print(f"  测试集: 范围=[{test_data.min():.4f}, {test_data.max():.4f}], 均值={test_data.mean():.4f}, 标准差={test_data.std():.4f}")
        
        # 计算分布差异
        mean_diff = abs(val_data.mean() - test_data.mean())
        std_diff = abs(val_data.std() - test_data.std())
        range_diff = abs((val_data.max() - val_data.min()) - (test_data.max() - test_data.min()))
        
        print(f"  差异: 均值差异={mean_diff:.4f}, 标准差差异={std_diff:.4f}, 范围差异={range_diff:.4f}")
        
        if mean_diff > 0.1 or std_diff > 0.1:
            print(f"  ⚠️ 数据分布差异较大！")
        else:
            print(f"  ✅ 数据分布相似")
    
    # 整体分析
    print(f"\n=== 整体分析 ===")
    val_mean = val_targets.mean(axis=0)
    test_mean = test_targets.mean(axis=0)
    val_std = val_targets.std(axis=0)
    test_std = test_targets.std(axis=0)
    
    print(f"验证集整体均值: {val_mean}")
    print(f"测试集整体均值: {test_mean}")
    print(f"验证集整体标准差: {val_std}")
    print(f"测试集整体标准差: {test_std}")
    
    mean_diff = np.abs(val_mean - test_mean).mean()
    std_diff = np.abs(val_std - test_std).mean()
    
    print(f"平均均值差异: {mean_diff:.4f}")
    print(f"平均标准差差异: {std_diff:.4f}")
    
    if mean_diff > 0.1 or std_diff > 0.1:
        print(f"\n❌ 验证集和测试集数据分布差异较大！")
        print(f"这可能是导致R²为负值的主要原因。")
        print(f"建议:")
        print(f"1. 检查数据划分是否合理")
        print(f"2. 重新随机划分数据集")
        print(f"3. 检查数据预处理是否一致")
    else:
        print(f"\n✅ 数据分布相似，问题可能在其他地方")

if __name__ == "__main__":
    main()

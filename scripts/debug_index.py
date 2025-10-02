"""
调试索引越界问题
找出哪个特征的索引超出范围
"""

import torch
import json
from pathlib import Path
from dataset import DesignLayoutDataset
from torch.utils.data import DataLoader
from dataset import collate_fn

# 加载数据
data_path = "/home/dell/Project-HCL/BaseLine/flexdm_pt/data/crello_json"
dataset = DesignLayoutDataset(data_path, split='test', max_length=20)

# 加载input_columns
with open('./input_columns_generated.json', 'r') as f:
    input_columns = json.load(f)

# 创建DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
example = next(iter(dataloader))

print("="*80)
print("检查每个特征的索引范围")
print("="*80)

errors = []

for key, value in example.items():
    if not torch.is_tensor(value):
        continue
    
    if key not in input_columns:
        print(f"\n⚠ {key}: 不在input_columns中")
        continue
    
    column = input_columns[key]
    
    if column['type'] == 'categorical':
        min_val = value.min().item()
        max_val = value.max().item()
        expected_max = column['input_dim'] + 1  # +2 for <MASK> and <NULL>
        
        print(f"\n{key}:")
        print(f"  实际范围: [{min_val}, {max_val}]")
        print(f"  input_dim: {column['input_dim']}")
        print(f"  嵌入层大小: {column['input_dim'] + 2} (含<MASK>和<NULL>)")
        
        if max_val >= column['input_dim'] + 2:
            error_msg = f"  ❌ 越界! {max_val} >= {column['input_dim'] + 2}"
            print(error_msg)
            errors.append((key, max_val, column['input_dim'] + 2))
        else:
            print(f"  ✓ 正常")

print("\n" + "="*80)
if errors:
    print("发现问题:")
    for key, max_val, vocab_size in errors:
        print(f"  {key}: 最大值{max_val} >= 词汇表大小{vocab_size}")
    
    print("\n修复建议:")
    print("1. 检查dataset.py中该特征的编码逻辑")
    print("2. 确保input_columns中的input_dim足够大")
    print("3. 可能需要在数据预处理时clip索引值")
else:
    print("✓ 所有特征索引都在正常范围内")

print("="*80)

# 详细检查问题特征
if errors:
    print("\n详细分析:")
    for key, _, _ in errors:
        print(f"\n{key} 的值分布:")
        value = example[key]
        unique_vals = torch.unique(value)
        print(f"  唯一值: {unique_vals.tolist()}")
        print(f"  数量: {len(unique_vals)}")
        
        # 检查数据集中的定义
        if hasattr(dataset, f'{key}_to_idx'):
            mapping = getattr(dataset, f'{key}_to_idx')
            print(f"  数据集映射大小: {len(mapping)}")
            print(f"  映射最大索引: {max(mapping.values())}")
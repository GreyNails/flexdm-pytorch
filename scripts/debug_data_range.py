"""
调试脚本：检查数据范围和模型输出维度的匹配
"""

import torch
import json
from pathlib import Path
from dataset import create_dataloader, DesignLayoutDataset
from models_pytorch import MFP

def check_data_ranges(data_dir, split='train', num_batches=10):
    """检查数据集中每个字段的实际值范围"""
    print("="*80)
    print(f"检查 {split} 数据集的值范围")
    print("="*80)
    
    dataset = DesignLayoutDataset(data_dir, split=split, max_length=20)
    loader = create_dataloader(data_dir, split, batch_size=32, shuffle=False, num_workers=0)
    
    # 收集统计信息
    stats = {}
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break
        
        for key, value in batch.items():
            if not torch.is_tensor(value) or key == 'id':
                continue
            
            if key not in stats:
                stats[key] = {
                    'min': float('inf'),
                    'max': float('-inf'),
                    'shape': value.shape,
                    'dtype': value.dtype
                }
            
            stats[key]['min'] = min(stats[key]['min'], value.min().item())
            stats[key]['max'] = max(stats[key]['max'], value.max().item())
    
    # 打印统计
    print("\n数据范围统计:")
    print("-" * 80)
    for key, info in sorted(stats.items()):
        print(f"{key:20s}: min={info['min']:8.2f}, max={info['max']:8.2f}, "
              f"shape={list(info['shape'])}, dtype={info['dtype']}")
    
    return stats, dataset


def check_model_output_dims(input_columns):
    """检查模型输出维度"""
    print("\n" + "="*80)
    print("模型输出维度配置")
    print("="*80)
    
    for key, config in input_columns.items():
        if not config.get('is_sequence', False):
            continue
        
        if config['type'] == 'categorical':
            shape = config.get('shape', [1])
            input_dim = config['input_dim']
            print(f"{key:20s}: type=categorical, input_dim={input_dim}, shape={shape}")
            print(f"  -> 预测类别范围: 0 到 {input_dim - 1}")
        elif config['type'] == 'numerical':
            shape = config.get('shape', [1])
            print(f"{key:20s}: type=numerical, shape={shape}")


def validate_data_model_match(stats, input_columns):
    """验证数据和模型配置是否匹配"""
    print("\n" + "="*80)
    print("验证数据和模型匹配性")
    print("="*80)
    
    issues = []
    
    for key, config in input_columns.items():
        if not config.get('is_sequence', False):
            continue
        if key not in stats:
            continue
        
        if config['type'] == 'categorical':
            input_dim = config['input_dim']
            max_val = stats[key]['max']
            min_val = stats[key]['min']
            
            # 检查范围
            if max_val >= input_dim:
                issue = f"❌ {key}: 数据最大值 {max_val} >= input_dim {input_dim}"
                issues.append(issue)
                print(issue)
                print(f"   数据范围: [{min_val}, {max_val}]")
                print(f"   模型期望: [0, {input_dim - 1}]")
            elif min_val < 0:
                issue = f"❌ {key}: 数据最小值 {min_val} < 0"
                issues.append(issue)
                print(issue)
            else:
                print(f"✓ {key}: 数据范围 [{min_val}, {max_val}] 符合模型期望 [0, {input_dim - 1}]")
    
    return issues


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                       default='/storage/HCL_data/crello_original/to_json')
    parser.add_argument('--config', type=str,
                       default='/home/dell/Project-HCL/BaseLine/flexdm_pt/scripts/config/train_config.json')
    args = parser.parse_args()
    
    # 1. 检查数据范围
    stats, dataset = check_data_ranges(args.data_dir, 'train', num_batches=20)
    
    # 2. 获取input_columns
    input_columns = dataset.get_input_columns()
    
    # 保存实际使用的配置
    with open('input_columns_actual.json', 'w') as f:
        json.dump(input_columns, f, indent=2)
    print(f"\n✓ 实际配置保存到: input_columns_actual.json")
    
    # 3. 检查模型输出维度
    check_model_output_dims(input_columns)
    
    # 4. 验证匹配性
    issues = validate_data_model_match(stats, input_columns)
    
    # 5. 总结
    print("\n" + "="*80)
    print("检查总结")
    print("="*80)
    
    if issues:
        print(f"\n发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  {issue}")
        print("\n建议修复方案:")
        print("  1. 调整 dataset.py 中的数据预处理逻辑")
        print("  2. 确保所有离散化的值都被 clip 到正确范围")
        print("  3. 检查 Encoder 中是否正确添加特殊 token")
    else:
        print("\n✓ 所有检查通过！数据和模型配置匹配。")
    
    # 6. 创建一个小批次测试前向传播
    print("\n" + "="*80)
    print("测试模型前向传播")
    print("="*80)
    
    try:
        model = MFP(input_columns, embed_dim=256, num_blocks=4, num_heads=8)
        model.eval()
        
        loader = create_dataloader(args.data_dir, 'train', batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        
        print("\n输入批次:")
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"  {key:20s}: shape={list(value.shape)}, "
                      f"min={value.min().item():.2f}, max={value.max().item():.2f}")
        
        with torch.no_grad():
            outputs = model(batch)
        
        print("\n模型输出:")
        for key, value in outputs.items():
            print(f"  {key:20s}: shape={list(value.shape)}")
        
        print("\n✓ 前向传播成功！")
        
    except Exception as e:
        print(f"\n✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
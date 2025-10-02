"""
TensorFlow Checkpoint 检查工具
用于在转换前检查checkpoint的结构和内容
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from collections import defaultdict


def inspect_checkpoint(checkpoint_path: str):
    """检查TensorFlow checkpoint的详细信息"""
    
    print("="*80)
    print(f"检查 TensorFlow Checkpoint")
    print(f"路径: {checkpoint_path}")
    print("="*80)
    
    # 加载checkpoint
    reader = tf.train.load_checkpoint(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    
    # 统计信息
    stats = {
        'total': 0,
        'model': 0,
        'optimizer': 0,
        'by_type': defaultdict(int),
        'by_module': defaultdict(int),
        'total_params': 0,
        'model_params': 0,
    }
    
    model_vars = []
    optimizer_vars = []
    special_vars = []
    
    print("\n" + "="*80)
    print("变量列表")
    print("="*80)
    
    for key in sorted(var_to_shape_map.keys()):
        stats['total'] += 1
        shape = var_to_shape_map[key]
        
        # 计算参数数量
        if shape:
            num_params = int(np.prod(shape))
            stats['total_params'] += num_params
        else:
            num_params = 1
        
        # 分类
        if 'optimizer' in key.lower():
            stats['optimizer'] += 1
            optimizer_vars.append((key, shape, num_params))
        elif '_CHECKPOINTABLE' in key:
            special_vars.append((key, shape, num_params))
        else:
            stats['model'] += 1
            stats['model_params'] += num_params
            model_vars.append((key, shape, num_params))
            
            # 按模块统计
            module = key.split('/')[0] if '/' in key else 'root'
            stats['by_module'][module] += 1
            
            # 按类型统计
            if 'kernel' in key.lower() or 'weight' in key.lower():
                stats['by_type']['weights'] += 1
            elif 'bias' in key.lower():
                stats['by_type']['biases'] += 1
            elif 'gamma' in key.lower() or 'beta' in key.lower():
                stats['by_type']['norm'] += 1
            elif 'embedding' in key.lower():
                stats['by_type']['embeddings'] += 1
            else:
                stats['by_type']['other'] += 1
    
    # 打印模型变量
    print("\n[模型权重]")
    print(f"共 {stats['model']} 个变量, {stats['model_params']:,} 个参数")
    print("-"*80)
    for key, shape, num_params in model_vars[:20]:  # 只显示前20个
        print(f"  {key:60s}  {str(shape):20s}  {num_params:>10,}")
    if len(model_vars) > 20:
        print(f"  ... 还有 {len(model_vars) - 20} 个变量")
    
    # 打印优化器变量
    print(f"\n[优化器状态]")
    print(f"共 {stats['optimizer']} 个变量")
    print("-"*80)
    for key, shape, num_params in optimizer_vars[:10]:  # 只显示前10个
        print(f"  {key:60s}  {str(shape):20s}")
    if len(optimizer_vars) > 10:
        print(f"  ... 还有 {len(optimizer_vars) - 10} 个变量")
    
    # 打印统计信息
    print("\n" + "="*80)
    print("统计摘要")
    print("="*80)
    print(f"总变量数:       {stats['total']:>6}")
    print(f"  - 模型权重:   {stats['model']:>6}  ({stats['model_params']:>12,} 参数)")
    print(f"  - 优化器:     {stats['optimizer']:>6}")
    print(f"  - 特殊标记:   {len(special_vars):>6}")
    
    print(f"\n按类型分类:")
    for type_name, count in sorted(stats['by_type'].items()):
        print(f"  - {type_name:15s}: {count:>6}")
    
    print(f"\n按模块分类:")
    for module, count in sorted(stats['by_module'].items(), key=lambda x: -x[1])[:10]:
        print(f"  - {module:30s}: {count:>6}")
    
    # 检查数据类型
    print("\n" + "="*80)
    print("数据类型检查")
    print("="*80)
    
    type_issues = []
    for key in list(var_to_shape_map.keys())[:5]:  # 检查前5个
        try:
            tensor = reader.get_tensor(key)
            dtype = tensor.dtype
            print(f"  ✓ {key[:50]:50s}  dtype={dtype}")
        except Exception as e:
            type_issues.append((key, str(e)))
            print(f"  ✗ {key[:50]:50s}  Error: {e}")
    
    if type_issues:
        print(f"\n⚠ 发现 {len(type_issues)} 个类型问题")
    
    # 推荐
    print("\n" + "="*80)
    print("转换建议")
    print("="*80)
    print(f"✓ 建议转换 {stats['model']} 个模型权重")
    print(f"✓ 建议跳过 {stats['optimizer']} 个优化器变量（训练时重新初始化）")
    print(f"✓ 预计转换后文件大小: ~{stats['model_params'] * 4 / (1024**2):.1f} MB")
    
    return stats


def compare_checkpoints(ckpt1: str, ckpt2: str):
    """比较两个checkpoint"""
    print("\n" + "="*80)
    print("比较 Checkpoints")
    print("="*80)
    
    reader1 = tf.train.load_checkpoint(ckpt1)
    reader2 = tf.train.load_checkpoint(ckpt2)
    
    vars1 = set(reader1.get_variable_to_shape_map().keys())
    vars2 = set(reader2.get_variable_to_shape_map().keys())
    
    only_in_1 = vars1 - vars2
    only_in_2 = vars2 - vars1
    common = vars1 & vars2
    
    print(f"\nCheckpoint 1: {ckpt1}")
    print(f"  变量数: {len(vars1)}")
    
    print(f"\nCheckpoint 2: {ckpt2}")
    print(f"  变量数: {len(vars2)}")
    
    print(f"\n共同变量: {len(common)}")
    
    if only_in_1:
        print(f"\n仅在Checkpoint 1中: {len(only_in_1)}")
        for var in list(only_in_1)[:5]:
            print(f"  - {var}")
        if len(only_in_1) > 5:
            print(f"  ... 还有 {len(only_in_1) - 5} 个")
    
    if only_in_2:
        print(f"\n仅在Checkpoint 2中: {len(only_in_2)}")
        for var in list(only_in_2)[:5]:
            print(f"  - {var}")
        if len(only_in_2) > 5:
            print(f"  ... 还有 {len(only_in_2) - 5} 个")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='检查TensorFlow checkpoint')
    parser.add_argument('checkpoint', help='Checkpoint路径')
    parser.add_argument('--compare', help='比较的第二个checkpoint路径')
    
    args = parser.parse_args()
    
    # 检查checkpoint
    stats = inspect_checkpoint(args.checkpoint)
    
    # 比较checkpoint（如果提供）
    if args.compare:
        compare_checkpoints(args.checkpoint, args.compare)


if __name__ == "__main__":
    # 示例用法
    CHECKPOINT_DIR = "/home/dell/Project-HCL/BaseLine/flex-dm/tmp/hc/checkpoints"
    
    print("检查 best checkpoint...")
    inspect_checkpoint(f"{CHECKPOINT_DIR}/best.ckpt")
    
    print("\n\n")
    print("检查 final checkpoint...")
    inspect_checkpoint(f"{CHECKPOINT_DIR}/final.ckpt")
    
    # 比较两个checkpoints
    print("\n\n")
    compare_checkpoints(
        f"{CHECKPOINT_DIR}/best.ckpt",
        f"{CHECKPOINT_DIR}/final.ckpt"
    )
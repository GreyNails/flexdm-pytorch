"""
配置调试工具
用于检查和修复input_columns配置问题
"""

import json
from pathlib import Path
from collections import Counter


def check_input_columns(input_columns: dict) -> dict:
    """
    检查input_columns配置
    
    Args:
        input_columns: 输入列配置字典
    
    Returns:
        检查结果
    """
    print("="*80)
    print("检查 input_columns 配置")
    print("="*80)
    
    issues = []
    warnings = []
    
    # 1. 检查重复的键
    keys = list(input_columns.keys())
    key_counts = Counter(keys)
    duplicates = {k: v for k, v in key_counts.items() if v > 1}
    
    if duplicates:
        issues.append(f"发现重复的键: {duplicates}")
        print("❌ 重复的键:")
        for key, count in duplicates.items():
            print(f"  - '{key}' 出现了 {count} 次")
    else:
        print("✓ 没有重复的键")
    
    # 2. 检查必需字段
    required_fields = ['is_sequence', 'type']
    for key, column in input_columns.items():
        missing = [f for f in required_fields if f not in column]
        if missing:
            issues.append(f"列 '{key}' 缺少字段: {missing}")
    
    if not issues:
        print("✓ 所有列都有必需字段")
    
    # 3. 检查类型一致性
    print("\n列类型统计:")
    seq_count = sum(1 for c in input_columns.values() if c.get('is_sequence', False))
    non_seq_count = len(input_columns) - seq_count
    print(f"  序列列: {seq_count}")
    print(f"  非序列列: {non_seq_count}")
    
    categorical_count = sum(1 for c in input_columns.values() 
                           if c.get('type') == 'categorical')
    numerical_count = sum(1 for c in input_columns.values() 
                         if c.get('type') == 'numerical')
    print(f"  分类特征: {categorical_count}")
    print(f"  数值特征: {numerical_count}")
    
    # 4. 检查维度配置
    print("\n维度配置:")
    for key, column in input_columns.items():
        if column.get('is_sequence', False):
            dim_info = f"  {key:20s}: "
            if column['type'] == 'categorical':
                dim_info += f"vocab_size={column.get('input_dim', '?')}"
            else:
                shape = column.get('shape', [1])
                dim_info += f"shape={shape}"
            print(dim_info)
    
    # 5. 警告检查
    for key, column in input_columns.items():
        if column['type'] == 'categorical' and 'input_dim' not in column:
            warnings.append(f"列 '{key}' 是分类类型但没有 'input_dim'")
        if column['type'] == 'numerical' and 'shape' not in column:
            warnings.append(f"列 '{key}' 是数值类型但没有 'shape'（将使用默认值[1]）")
    
    if warnings:
        print("\n⚠ 警告:")
        for w in warnings:
            print(f"  - {w}")
    
    print("\n" + "="*80)
    if issues:
        print(f"❌ 发现 {len(issues)} 个问题")
        for issue in issues:
            print(f"  - {issue}")
        return {'status': 'error', 'issues': issues}
    elif warnings:
        print(f"⚠ 发现 {len(warnings)} 个警告")
        return {'status': 'warning', 'warnings': warnings}
    else:
        print("✓ 配置检查通过")
        return {'status': 'ok'}


def create_input_columns_from_data(data_path: str, split: str = 'train') -> dict:
    """
    从数据文件自动创建input_columns
    
    Args:
        data_path: 数据路径
        split: 数据集划分
    
    Returns:
        input_columns配置
    """
    json_file = Path(data_path) / f"{split}.json"
    
    if not json_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {json_file}")
    
    print(f"从 {json_file} 推断配置...")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        raise ValueError("数据文件为空")
    
    sample = data[0]
    input_columns = {}
    
    # 非序列特征
    for key in ['canvas_width', 'canvas_height', 'length']:
        if key in sample:
            input_columns[key] = {
                'is_sequence': False,
                'type': 'categorical',
                'input_dim': 2000,  # 足够大的值
            }
    
    # 序列特征
    if 'elements' not in sample or not sample['elements']:
        # 直接从顶层推断
        for key, value in sample.items():
            if key in ['id', 'canvas_width', 'canvas_height', 'length']:
                continue
            
            if isinstance(value, list) and value:
                first_elem = value[0]
                
                if isinstance(first_elem, (int, str)):
                    # 分类特征
                    unique_values = set()
                    for item in data[:100]:  # 检查前100个样本
                        if key in item:
                            unique_values.update(item[key])
                    
                    input_columns[key] = {
                        'is_sequence': True,
                        'type': 'categorical',
                        'input_dim': len(unique_values) + 10,  # 留出余量
                        'shape': [1],
                    }
                
                elif isinstance(first_elem, float):
                    # 数值特征（可能需要离散化）
                    input_columns[key] = {
                        'is_sequence': True,
                        'type': 'categorical',
                        'input_dim': 64,  # 位置特征通常离散化为64个bins
                        'shape': [1],
                    }
                
                elif isinstance(first_elem, list):
                    # 向量特征
                    input_columns[key] = {
                        'is_sequence': True,
                        'type': 'numerical',
                        'shape': [len(first_elem)],
                    }
    else:
        # 从elements推断
        elements = sample['elements']
        if elements:
            first_elem = elements[0]
            
            for key, value in first_elem.items():
                if isinstance(value, str):
                    # 字符串 -> 分类
                    unique_values = set()
                    for item in data[:100]:
                        for elem in item.get('elements', []):
                            if key in elem:
                                unique_values.add(elem[key])
                    
                    input_columns[key] = {
                        'is_sequence': True,
                        'type': 'categorical',
                        'input_dim': max(len(unique_values) + 5, 10),
                        'shape': [1],
                    }
                
                elif isinstance(value, (int, float)):
                    # 数值 -> 离散化为分类
                    input_columns[key] = {
                        'is_sequence': True,
                        'type': 'categorical',
                        'input_dim': 64,
                        'shape': [1],
                    }
                
                elif isinstance(value, list):
                    if all(isinstance(v, (int, float)) for v in value):
                        # 数值向量
                        if len(value) == 3:  # RGB
                            input_columns[key] = {
                                'is_sequence': True,
                                'type': 'categorical',
                                'input_dim': 16,  # 颜色通常离散化
                                'shape': [3],
                            }
                        else:  # 嵌入向量
                            input_columns[key] = {
                                'is_sequence': True,
                                'type': 'numerical',
                                'shape': [len(value)],
                            }
    
    print(f"\n✓ 推断出 {len(input_columns)} 个特征")
    
    return input_columns


def fix_input_columns(input_columns: dict) -> dict:
    """
    修复input_columns中的常见问题
    
    Args:
        input_columns: 原始配置
    
    Returns:
        修复后的配置
    """
    print("修复配置...")
    
    # 去重
    seen = set()
    fixed = {}
    
    for key, column in input_columns.items():
        if key in seen:
            print(f"  跳过重复的键: {key}")
            continue
        seen.add(key)
        
        # 添加缺失的默认值
        if 'is_sequence' not in column:
            column['is_sequence'] = True
            print(f"  为 {key} 添加默认 is_sequence=True")
        
        if 'type' not in column:
            # 根据其他字段推断类型
            if 'input_dim' in column:
                column['type'] = 'categorical'
            elif 'shape' in column:
                column['type'] = 'numerical'
            else:
                column['type'] = 'categorical'
                column['input_dim'] = 10
            print(f"  为 {key} 添加推断的 type={column['type']}")
        
        # 确保分类特征有input_dim
        if column['type'] == 'categorical' and 'input_dim' not in column:
            column['input_dim'] = 10
            print(f"  为 {key} 添加默认 input_dim=10")
        
        # 确保数值特征有shape
        if column['type'] == 'numerical' and 'shape' not in column:
            column['shape'] = [1]
            print(f"  为 {key} 添加默认 shape=[1]")
        
        fixed[key] = column
    
    print(f"✓ 修复完成，保留 {len(fixed)} 个特征")
    return fixed


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='配置调试工具')
    parser.add_argument('--data_path', type=str, 
                       default='./data/crello_json',
                       help='数据路径')
    parser.add_argument('--action', choices=['check', 'create', 'fix'],
                       default='check',
                       help='操作: check(检查), create(创建), fix(修复)')
    
    args = parser.parse_args()
    
    if args.action == 'create':
        # 从数据创建配置
        input_columns = create_input_columns_from_data(args.data_path)
        
        # 检查
        result = check_input_columns(input_columns)
        
        # 保存
        output_file = 'input_columns_generated.json'
        with open(output_file, 'w') as f:
            json.dump(input_columns, f, indent=2)
        print(f"\n✓ 配置已保存到: {output_file}")
    
    elif args.action == 'fix':
        # 加载现有配置
        config_file = 'input_columns.json'
        if not Path(config_file).exists():
            print(f"配置文件不存在: {config_file}")
            return
        
        with open(config_file, 'r') as f:
            input_columns = json.load(f)
        
        # 修复
        fixed = fix_input_columns(input_columns)
        
        # 检查
        result = check_input_columns(fixed)
        
        # 保存
        output_file = 'input_columns_fixed.json'
        with open(output_file, 'w') as f:
            json.dump(fixed, f, indent=2)
        print(f"\n✓ 修复的配置已保存到: {output_file}")
    
    else:  # check
        # 示例配置检查
        test_config = {
            'type': {'is_sequence': True, 'type': 'categorical', 'input_dim': 7, 'shape': [1]},
            'left': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
            'top': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
            'width': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
            'height': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
            'image_embedding': {'is_sequence': True, 'type': 'numerical', 'shape': [512]},
        }
        
        check_input_columns(test_config)


if __name__ == "__main__":
    main()
"""
验证PyTorch配置与TensorFlow原版是否对齐
"""

import json
from pathlib import Path

# TensorFlow原版的input_columns（从用户提供的数据）
TF_INPUT_COLUMNS = {
    "type": {"input_dim": 6},
    "left": {"input_dim": 64},
    "top": {"input_dim": 64},
    "width": {"input_dim": 64},
    "height": {"input_dim": 64},
    "opacity": {"input_dim": 8},
    "color": {"input_dim": 16, "shape": [3]},
    "image_embedding": {"shape": [512]},
    "text_embedding": {"shape": [512]},
    "font_family": {"input_dim": 35},
}

# TensorFlow原版的模型结构（从用户提供的打印）
TF_DECODER_SHAPES = {
    "type": (256, 6),
    "left": (256, 64),
    "top": (256, 64),
    "width": (256, 64),
    "height": (256, 64),
    "opacity": (256, 8),
    "color": (256, 48),  # 3 * 16
    "image_embedding": (256, 512),
    "text_embedding": (256, 512),
    "font_family": (256, 35),
}

print("="*80)
print("配置对齐验证")
print("="*80)

# 步骤1：生成PyTorch的input_columns
print("\n步骤1: 生成PyTorch input_columns")
from dataset import DesignLayoutDataset

dataset = DesignLayoutDataset(
    '/home/dell/Project-HCL/BaseLine/flexdm_pt/data/crello_json',
    split='test',
    max_length=20
)

pt_input_columns = dataset.get_input_columns()

# 保存
with open('input_columns_verified.json', 'w') as f:
    json.dump(pt_input_columns, f, indent=2)
print("✓ 已保存到 input_columns_verified.json")

# 步骤2：验证关键字段
print("\n步骤2: 验证input_dim")
print("-"*80)
print(f"{'字段':<20} {'TF原版':<15} {'PyTorch':<15} {'状态'}")
print("-"*80)

all_match = True
for key in TF_INPUT_COLUMNS.keys():
    if 'embedding' in key:
        continue  # numerical类型跳过
    
    tf_dim = TF_INPUT_COLUMNS[key]['input_dim']
    pt_dim = pt_input_columns[key]['input_dim']
    
    match = "✓" if tf_dim == pt_dim else "✗"
    if tf_dim != pt_dim:
        all_match = False
    
    print(f"{key:<20} {tf_dim:<15} {pt_dim:<15} {match}")

print("-"*80)

# 步骤3：验证Encoder层大小
print("\n步骤3: 验证Encoder Embedding层大小")
print("-"*80)
print(f"{'字段':<20} {'预期大小':<15} {'说明'}")
print("-"*80)

for key in TF_INPUT_COLUMNS.keys():
    if 'embedding' in key:
        continue
    
    expected_size = TF_INPUT_COLUMNS[key]['input_dim'] + 2  # +2 for <MASK> and <NULL>
    print(f"{key:<20} {expected_size:<15} Embedding({expected_size}, 256)")

print("-"*80)

# 步骤4：验证Decoder输出维度
print("\n步骤4: 验证Decoder输出维度")
print("-"*80)
print(f"{'字段':<20} {'TF输出':<20} {'PyTorch应该':<20} {'状态'}")
print("-"*80)

for key, (in_dim, out_dim) in TF_DECODER_SHAPES.items():
    if key not in pt_input_columns:
        continue
    
    column = pt_input_columns[key]
    
    if column['type'] == 'categorical':
        shape = column.get('shape', [1])
        expected_out = shape[-1] * column['input_dim']
        pt_out = f"{shape[-1]} * {column['input_dim']} = {expected_out}"
        
        match = "✓" if expected_out == out_dim else "✗"
        if expected_out != out_dim:
            all_match = False
    else:
        expected_out = column['shape'][-1]
        pt_out = str(expected_out)
        match = "✓" if expected_out == out_dim else "✗"
        if expected_out != out_dim:
            all_match = False
    
    print(f"{key:<20} {out_dim:<20} {pt_out:<20} {match}")

print("-"*80)

# 步骤5：创建模型验证结构
print("\n步骤5: 创建模型验证")
from models_pytorch import MFP

model = MFP(
    input_columns=pt_input_columns,
    embed_dim=256,
    num_blocks=4,
    num_heads=8,
    dropout=0.1,
)

print("\n" + "="*80)
if all_match:
    print("✓ 所有配置与TensorFlow原版一致！")
    print("\n可以安全地加载转换后的权重:")
    print("  checkpoint = torch.load('best_pytorch.pth')")
    print("  model.load_state_dict(checkpoint['state_dict'], strict=False)")
else:
    print("✗ 发现不一致，请检查上面标记为✗的项")
    print("\n不建议加载权重，可能导致：")
    print("  1. 权重无法加载（维度不匹配）")
    print("  2. 预测结果错误（索引映射不对）")
print("="*80)
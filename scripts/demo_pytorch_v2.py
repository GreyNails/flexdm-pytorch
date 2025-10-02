"""
PyTorch版本的MFP模型演示和可视化
用于测试和可视化布局生成结果
"""
import json
import itertools
import logging
from pathlib import Path
from typing import Dict, List
import json

import torch
import numpy as np
from IPython.display import display, HTML

# 导入PyTorch模型和工具
# from models_pytorch import MFP
from dataset import DesignLayoutDataset
from svg_builder_pytorch import SVGBuilder
from retriever_pytorch import ImageRetriever, TextRetriever

import sys
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)

class FixedEncoder(nn.Module):
    def __init__(self, input_columns, embed_dim=128, dropout=0.1, max_length=50):
        super().__init__()
        self.input_columns = input_columns
        self.embed_dim = embed_dim
        
        # 使用列表存储，避免ModuleDict的键冲突
        self.emb_layers = nn.ModuleList()
        self.emb_keys = []
        self.emb_types = []
        
        for key, column in input_columns.items():
            if not column.get('is_sequence', False):
                continue
            
            self.emb_keys.append(key)
            self.emb_types.append(column['type'])
            
            if column['type'] == 'categorical':
                vocab_size = column['input_dim'] + 2
                self.emb_layers.append(nn.Embedding(vocab_size, embed_dim))
            else:
                input_size = column.get('shape', [1])[-1]
                self.emb_layers.append(nn.Linear(input_size, embed_dim))
        
        self.pos_embedding = nn.Embedding(max_length + 1, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs):
        batch_size = inputs['length'].size(0)
        seq_len = inputs[self.emb_keys[0]].size(1)
        
        seq_embs = []
        for idx, (key, typ) in enumerate(zip(self.emb_keys, self.emb_types)):
            if key in inputs:
                x = inputs[key]
                emb = self.emb_layers[idx](x)
                if emb.dim() == 4:
                    emb = emb.sum(dim=2)
                seq_embs.append(emb)
        
        seq = torch.stack(seq_embs).sum(dim=0)
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        seq = seq + self.pos_embedding(positions)
        seq = self.dropout(seq)
        
        lengths = inputs['length'].squeeze(-1)
        mask = torch.arange(seq_len, device=seq.device).unsqueeze(0) < lengths.unsqueeze(1)
        return seq, mask

print("应用临时修复...")
import models_pytorch
models_pytorch.Encoder = FixedEncoder
print("✓ Encoder已替换")
from models_pytorch import MFP


class DemoConfig:
    """演示配置"""
    def __init__(self):
        self.ckpt_dir = "/home/dell/Project-HCL/BaseLine/flexdm_pt/chechpoints"
        self.dataset_name = "crello"
        self.db_root = "/home/dell/Project-HCL/BaseLine/flexdm_pt/data/crello_json"
        self.batch_size = 20
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 任务类型: elem, pos, attr, txt, img
        self.target_task = "pos"
        
        # 列名配置
        self.column_names = {
            "txt": ["gt-layout", "gt-visual", "input", "pred"],
            "img": ["gt-layout", "gt-visual", "input", "pred"],
            "attr": ["gt-layout", "gt-visual", "input", "pred"],
            "pos": ["gt-layout", "gt-visual", "pred-layout", "pred-visual"],
            "elem": ["gt-layout", "gt-visual", "input-layout", "input-visual", "pred-layout", "pred-visual"],
        }
        
        # 属性分组
        self.attribute_groups = {
            "type": ["type"],
            "pos": ["left", "top", "width", "height"],
            "attr": ["opacity", "color", "font_family"],
            "img": ["image_embedding"],
            "txt": ["text_embedding"],
        }


def load_model(checkpoint_path: str, input_columns: Dict, device: str = 'cuda'):
    """
    加载PyTorch模型
    
    Args:
        checkpoint_path: checkpoint路径
        input_columns: 输入列配置
        device: 设备
    
    Returns:
        加载好的模型
    """
    # 创建模型
    model = MFP(
        input_columns=input_columns,
        embed_dim=256,
        num_blocks=4,
        num_heads=8,
        dropout=0.1,
    )
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 加载权重（允许部分匹配）
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        logger.warning(f"Missing keys: {len(missing)}")
    if unexpected:
        logger.warning(f"Unexpected keys: {len(unexpected)}")
    
    model.to(device)
    model.eval()
    
    logger.info(f"✓ Model loaded from {checkpoint_path}")
    return model


def get_seq_mask(lengths: torch.Tensor, max_len: int = None) -> torch.Tensor:
    """
    生成序列掩码
    
    Args:
        lengths: (B,) 长度张量
        max_len: 最大长度
    
    Returns:
        mask: (B, S) 布尔掩码
    """
    if lengths.dim() == 2:
        lengths = lengths.squeeze(-1)
    
    batch_size = lengths.size(0)
    if max_len is None:
        max_len = lengths.max().item()
    
    # 创建掩码
    mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    return mask


def get_initial_masks(input_columns: Dict, seq_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    初始化掩码字典（所有为False）
    
    Args:
        input_columns: 输入列配置
        seq_mask: 序列掩码
    
    Returns:
        masks: 掩码字典
    """
    masks = {}
    batch_size, seq_len = seq_mask.shape
    
    for key, column in input_columns.items():
        if column.get('is_sequence', False):
            masks[key] = torch.zeros_like(seq_mask, dtype=torch.bool)
        else:
            masks[key] = torch.ones(batch_size, dtype=torch.bool)
    
    return masks


def set_visual_default(item: Dict) -> Dict:
    """设置可视化默认值"""
    item = item.copy()
    for elem in item.get('elements', []):
        if 'color' not in elem or elem['color'] is None:
            elem['color'] = [0, 0, 0]
        if 'opacity' not in elem or elem['opacity'] is None:
            elem['opacity'] = 1.0
        if 'font_family' not in elem or elem['font_family'] is None:
            elem['font_family'] = 'DummyFont'
    return item


def tensor_to_list(data: Dict) -> List[Dict]:
    """
    将批次张量转换为样本列表
    
    Args:
        data: 批次数据字典
    
    Returns:
        样本列表
    """
    batch_size = data['length'].size(0)
    items = []
    
    for i in range(batch_size):
        item = {
            'id': data['id'][i] if 'id' in data else f'sample_{i}',
            'canvas_width': data['canvas_width'][i].item(),
            'canvas_height': data['canvas_height'][i].item(),
            'length': data['length'][i].item(),
            'elements': []
        }
        
        # 获取有效长度
        length = item['length'] + 1  # 基于0的索引
        
        # 构建元素列表
        for j in range(length):
            element = {}
            
            for key, value in data.items():
                if key in ['id', 'length', 'canvas_width', 'canvas_height']:
                    continue
                
                if value.dim() >= 2 and value.size(1) > j:
                    elem_value = value[i, j]
                    
                    # 转换为Python原生类型
                    if torch.is_tensor(elem_value):
                        if elem_value.dim() == 0:
                            element[key] = elem_value.item()
                        elif elem_value.dim() == 1:
                            element[key] = elem_value.cpu().numpy().tolist()
                        else:
                            # 对于分类变量，取argmax
                            if elem_value.dim() == 2:
                                element[key] = elem_value.argmax(dim=-1).cpu().numpy().tolist()
                            else:
                                element[key] = elem_value.cpu().numpy().tolist()
            
            item['elements'].append(element)
        
        items.append(item)
    
    return items


def apply_task_masks(
    example: Dict,
    input_columns: Dict,
    target_task: str,
    attribute_groups: Dict,
    device: str
) -> Dict[str, torch.Tensor]:
    """
    应用任务特定的掩码
    
    Args:
        example: 输入样本
        input_columns: 输入列配置
        target_task: 目标任务
        attribute_groups: 属性分组
        device: 设备
    
    Returns:
        masks: 掩码字典
    """
    seq_mask = get_seq_mask(example['length'], example['left'].size(1))
    mfp_masks = get_initial_masks(input_columns, seq_mask)
    
    for key in mfp_masks.keys():
        if not input_columns[key].get('is_sequence', False):
            continue
        
        mask = mfp_masks[key].clone()
        
        if target_task == "elem":
            # 元素级掩码：隐藏第一个元素
            mask[:, 0] = True
        else:
            # 特征级掩码
            if key == "type":
                continue
            
            if target_task in attribute_groups:
                attr_keys = attribute_groups[target_task]
                if key in attr_keys:
                    mask = seq_mask.clone()
        
        mfp_masks[key] = mask.to(device)
    
    return mfp_masks


def visualize_reconstruction(
    model: torch.nn.Module,
    example: Dict,
    builders: Dict,
    config: DemoConfig,
    input_columns: Dict,
):
    """
    可视化重建结果
    
    Args:
        model: PyTorch模型
        example: 输入样本
        builders: SVG构建器字典
        config: 配置
        input_columns: 输入列配置
    
    Returns:
        SVG列表
    """
    svgs = []
    target_task = config.target_task
    
    # 转换为样本列表
    items = tensor_to_list(example)
    
    # GT布局和视觉
    svgs.append(list(map(builders["layout"], items)))
    svgs.append(list(map(builders["visual"], items)))
    
    # 输入可视化（根据任务类型）
    if target_task == "txt":
        svgs.append(list(map(builders["visual_wo_text"], items)))
    elif target_task == "img":
        svgs.append(list(map(builders["visual_wo_image"], items)))
    elif target_task == "attr":
        svgs.append(list(map(builders["visual"], [set_visual_default(x) for x in items])))
    
    # 应用掩码
    mfp_masks = apply_task_masks(
        example, input_columns, target_task, 
        config.attribute_groups, config.device
    )
    
    # 元素级任务的特殊处理
    if target_task == "elem":
        # 创建移除第一个元素后的样本
        example_copy = {}
        for key, value in example.items():
            if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.size(1) > 1:
                # 移除第一个元素
                indices = torch.where(~mfp_masks[key][0, :])[0]
                example_copy[key] = torch.index_select(value, 1, indices)
            else:
                example_copy[key] = value
        
        example_copy['length'] = example['length'] - 1
        
        items_copy = tensor_to_list(example_copy)
        svgs.append(list(map(builders["layout"], items_copy)))
        svgs.append(list(map(builders["visual"], items_copy)))
    
    # 模型预测
    with torch.no_grad():
        # 将掩码信息添加到输入
        pred = model_inference_with_masks(model, example, mfp_masks)
    
    # 合并预测和原始输入
    for key in example:
        if key not in pred:
            pred[key] = example[key]
    
    # 预测可视化
    pred_items = tensor_to_list(pred)
    
    if target_task in ["pos", "elem"]:
        svgs.append(list(map(builders["layout"], pred_items)))
    svgs.append(list(map(builders["visual"], pred_items)))
    
    return [list(grouper(row, len(config.column_names[target_task]))) for row in zip(*svgs)]


def model_inference_with_masks(model, inputs, masks):
    """
    使用掩码进行模型推理
    
    Args:
        model: 模型
        inputs: 输入数据
        masks: 掩码字典
    
    Returns:
        预测结果
    """
    # 应用掩码到输入
    masked_inputs = {}
    for key, value in inputs.items():
        if key in masks and torch.is_tensor(value):
            mask = masks[key]
            if mask.any():
                # 应用掩码（使用特殊token）
                masked_value = value.clone()
                if value.dim() == 3:  # (B, S, F)
                    masked_value[mask] = 0  # 或使用特殊值
                masked_inputs[key] = masked_value
            else:
                masked_inputs[key] = value
        else:
            masked_inputs[key] = value
    
    # 模型推理
    outputs = model(masked_inputs)
    
    return outputs


def grouper(iterable, n):
    """将可迭代对象分组"""
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=None)


def main():
    """主函数"""
    # 配置
    config = DemoConfig()
    
    logger.info("="*80)
    logger.info("MFP PyTorch Demo")
    logger.info("="*80)
    
    # 加载数据
    logger.info(f"Loading dataset from {config.db_root}")
    dataset = DesignLayoutDataset(
        config.db_root, 
        split='test',
        max_length=50
    )
    
    # 创建DataLoader
    from torch.utils.data import DataLoader
    from dataset import collate_fn

    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 获取一个批次
    example = next(iter(dataloader))
    
    # 移动到设备
    for key in example:
        if torch.is_tensor(example[key]):
            example[key] = example[key].to(config.device)

    with open('/home/dell/Project-HCL/BaseLine/flexdm_pt/scripts/input_columns_generated.json', 'r') as f:
        input_columns = json.load(f)    
    
    # # 获取输入列配置
    # input_columns = {
    #     'type': {'is_sequence': True, 'type': 'categorical', 'input_dim': 7, 'shape': [1]},
    #     'left': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
    #     'top': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
    #     'width': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
    #     'height': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
    #     'image_embedding': {'is_sequence': True, 'type': 'numerical', 'shape': [512]},
    # }
    
    # 加载模型
    logger.info(f"Loading model from {config.ckpt_dir}")
    checkpoint_path = Path(config.ckpt_dir) / "best_pytorch.pth"
    model = load_model(str(checkpoint_path), input_columns, config.device)
    
    # 构建检索数据库
    logger.info("Building retrieval databases...")
    db_root = Path(config.db_root).parent / config.dataset_name
    
    image_db = ImageRetriever(db_root, image_path=db_root / "images")
    image_db.build("test")
    
    text_db = TextRetriever(db_root, text_path=db_root / "texts")
    text_db.build("test")
    
    # 创建SVG构建器
    logger.info("Creating SVG builders...")
    builders = {}
    
    # 布局构建器
    builders["layout"] = SVGBuilder(
        max_width=128,
        max_height=192,
        key="type",
    )
    
    # 视觉构建器
    patterns = [
        ("visual", image_db, text_db),
        ("visual_wo_text", image_db, None),
        ("visual_wo_image", None, text_db),
    ]
    
    for (name, idb, tdb) in patterns:
        builders[name] = SVGBuilder(
            max_width=128,
            max_height=192,
            key="color",
            image_db=idb,
            text_db=tdb,
            render_text=True,
        )
    
    # 可视化重建
    logger.info(f"Visualizing reconstruction for task: {config.target_task}")
    logger.info(f"Columns: {', '.join(config.column_names[config.target_task])}")
    
    svgs = visualize_reconstruction(
        model, example, builders, config, input_columns
    )
    
    # 显示结果
    for i, row in enumerate(svgs):
        print(f"Sample {i}:")
        display(HTML("<div>%s</div>" % " ".join(itertools.chain.from_iterable(row))))
    
    logger.info("✓ Demo completed!")


if __name__ == "__main__":
    main()
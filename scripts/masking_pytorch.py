"""
PyTorch版本的Masking机制
严格对齐TensorFlow原始实现
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np

# 特殊值常量（与TF版本一致）
MASK_VALUE = 10.0
NULL_VALUE = 0.0

# Masking概率（与TF版本一致）
MASK_PROB = 0.15
REPLACE_PROB = 0.1
UNCHANGE_PROB = 0.1
CHANGE_PROB = 1.0 - UNCHANGE_PROB
THRESH = REPLACE_PROB / CHANGE_PROB


def get_task_names(input_columns: Dict) -> List[str]:
    """获取任务名称列表"""
    task_names = ["random", "elem"]
    
    # 从input_columns推断属性组
    attribute_groups = get_attribute_groups(input_columns.keys())
    task_names += list(attribute_groups.keys())
    
    return task_names


def get_attribute_groups(keys: List[str]) -> Dict[str, List[str]]:
    """获取属性组定义"""
    # Crello数据集的属性组
    if 'font_family' in keys or 'opacity' in keys:
        return {
            'type': ['type'],
            'pos': ['left', 'top', 'width', 'height'],
            'attr': ['opacity', 'color', 'font_family'],
            'img': ['image_embedding'],
            'txt': ['text_embedding'],
        }
    # Rico数据集的属性组
    else:
        return {
            'type': ['type'],
            'pos': ['left', 'top', 'width', 'height'],
            'attr': ['icon', 'clickable', 'text_button'],
        }


def get_seq_mask(length: torch.Tensor, max_len: int = None) -> torch.Tensor:
    """
    从长度生成序列mask
    
    Args:
        length: (B, 1) 或 (B,) - 序列长度（zero-based）
        max_len: 最大长度
    
    Returns:
        mask: (B, S) - True表示有效位置
    """
    if length.dim() == 2:
        length = length.squeeze(-1)
    
    # 转换为one-based
    length = length + 1
    
    if max_len is None:
        max_len = length.max().item()
    
    # 生成mask
    batch_size = length.size(0)
    mask = torch.arange(max_len, device=length.device).unsqueeze(0).expand(batch_size, -1)
    mask = mask < length.unsqueeze(1)
    
    return mask


def apply_token(
    input_tensor: torch.Tensor,
    column: Dict,
    mask: torch.Tensor,
    token_type: str
) -> torch.Tensor:
    """
    应用特殊token（MASK/UNUSED/RANDOM）
    
    Args:
        input_tensor: 输入张量
        column: 列配置
        mask: (B, S) mask，True表示需要替换的位置
        token_type: 'masked', 'unused', 或 'random'
    
    Returns:
        处理后的张量
    """
    assert token_type in ['masked', 'unused', 'random']
    
    # 扩展mask维度以匹配input
    while mask.dim() < input_tensor.dim():
        mask = mask.unsqueeze(-1)
    
    if column['type'] == 'categorical':
        # 分类特征
        if token_type == 'masked':
            # MASK token = input_dim
            value = torch.full_like(input_tensor, column['input_dim'])
        elif token_type == 'unused':
            # UNUSED token = input_dim + 1
            value = torch.full_like(input_tensor, column['input_dim'] + 1)
        else:  # random
            value = torch.randint_like(input_tensor, 0, column['input_dim'])
        
        output = torch.where(mask, value, input_tensor)
    
    else:  # numerical
        # 数值特征
        if token_type == 'masked':
            value = torch.full_like(input_tensor, MASK_VALUE)
        elif token_type == 'unused':
            value = torch.full_like(input_tensor, NULL_VALUE)
        else:  # random
            value = torch.randn_like(input_tensor) * 0.1
        
        output = torch.where(mask, value, input_tensor)
    
    return output


def get_initial_masks(input_columns: Dict, seq_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    初始化mask字典（全False）
    
    Args:
        input_columns: 列配置
        seq_mask: (B, S) 序列mask
    
    Returns:
        masks: 每个特征的mask（全False）
    """
    masks = {}
    batch_size = seq_mask.size(0)
    
    for key, column in input_columns.items():
        if not column.get('is_sequence', False):
            # 非序列特征 - 全True（表示需要预测）
            masks[key] = torch.ones(batch_size, dtype=torch.bool, device=seq_mask.device)
        else:
            # 序列特征 - 全False
            masks[key] = torch.zeros_like(seq_mask, dtype=torch.bool)
    
    return masks


def filter_padding(
    inputs: Dict[str, torch.Tensor],
    input_columns: Dict,
    mask: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    将padding位置设置为NULL
    
    Args:
        inputs: 输入字典
        input_columns: 列配置
        mask: (B, S) 有效位置mask
    
    Returns:
        处理后的输入
    """
    modified_inputs = {}
    unused_mask = ~mask  # padding位置
    
    for key, column in input_columns.items():
        # ⭐ 跳过demo_only字段（如uuid）
        if column.get('demo_only', False):
            modified_inputs[key] = inputs.get(key, None)
            continue
        
        # ⭐ 跳过不存在的字段
        if key not in inputs:
            continue
        
        input_val = inputs[key]
        
        # ⭐ 跳过非tensor类型（安全检查）
        if not torch.is_tensor(input_val):
            modified_inputs[key] = input_val
            continue
        
        if column.get('is_sequence', False):
            # 检查loss_condition
            if 'loss_condition' in column:
                cond = column['loss_condition']
                # 某些类型的元素不需要这个特征
                cond_mask = torch.zeros_like(mask, dtype=torch.bool)
                
                # 获取条件字段的值
                if cond['key'] in inputs and torch.is_tensor(inputs[cond['key']]):
                    type_values = inputs[cond['key']]  # (B, S, 1)
                    if type_values.dim() == 3:
                        type_values = type_values.squeeze(-1)  # (B, S)
                    
                    # 根据loss_condition的mask设置
                    if 'mask' in cond:
                        # cond['mask']是一个列表，指示每个类别是否需要这个特征
                        loss_mask_tensor = torch.tensor(
                            cond['mask'], 
                            dtype=torch.bool, 
                            device=type_values.device
                        )
                        # 使用gather获取每个位置对应的mask值
                        # 确保type_values在有效范围内
                        type_values = torch.clamp(type_values, 0, len(cond['mask']) - 1)
                        cond_mask = ~loss_mask_tensor[type_values]
                
                mask_ = cond_mask | unused_mask
            else:
                mask_ = unused_mask
            
            # 应用UNUSED token
            modified_inputs[key] = apply_token(input_val, column, mask_, 'unused')
        else:
            modified_inputs[key] = input_val
    
    return modified_inputs



def random_masking(
    inputs: Dict[str, torch.Tensor],
    input_columns: Dict,
    mask: torch.Tensor
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    随机masking（MLM风格）
    - 15%的token被选中
    - 选中的token中：80% MASK, 10% RANDOM, 10% UNCHANGED
    
    Args:
        inputs: 输入字典
        input_columns: 列配置  
        mask: (B, S) 有效位置mask
    
    Returns:
        (modified_inputs, mfp_masks)
    """
    modified_inputs = {}
    mfp_masks = {}
    
    for key, column in input_columns.items():
        # ⭐ 跳过demo_only字段
        if column.get('demo_only', False):
            modified_inputs[key] = inputs.get(key, None)
            mfp_masks[key] = torch.zeros(mask.size(0), dtype=torch.bool, device=mask.device)
            continue
        
        # ⭐ 跳过不存在的字段
        if key not in inputs:
            continue
        
        if not column.get('is_sequence', False):
            modified_inputs[key] = inputs[key]
            mfp_masks[key] = torch.ones(
                mask.size(0), dtype=torch.bool, device=mask.device
            )
            continue
        
        input_val = inputs[key]
        
        # ⭐ 跳过非tensor
        if not torch.is_tensor(input_val):
            modified_inputs[key] = input_val
            mfp_masks[key] = torch.zeros_like(mask, dtype=torch.bool)
            continue
        
        # 随机选择MASK_PROB比例的位置
        rand_arr = torch.rand(mask.shape, device=mask.device)
        mfp_mask = mask & (rand_arr < MASK_PROB)
        
        # 在选中的位置中，80% MASK, 10% RANDOM, 10% UNCHANGED
        chg_mask = mfp_mask & (torch.rand_like(rand_arr) < CHANGE_PROB)
        rand_arr2 = torch.rand_like(rand_arr)
        
        masked_input = apply_token(input_val, column, chg_mask & (rand_arr2 >= THRESH), 'masked')
        masked_input = apply_token(masked_input, column, chg_mask & (rand_arr2 < THRESH), 'random')
        
        modified_inputs[key] = masked_input
        mfp_masks[key] = mfp_mask
    
    return modified_inputs, mfp_masks

def elem_masking(
    inputs: Dict[str, torch.Tensor],
    input_columns: Dict,
    mask: torch.Tensor,
    is_autoreg: bool = False
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    元素级masking（mask整个元素的所有特征）
    
    Args:
        inputs: 输入字典
        input_columns: 列配置
        mask: (B, S) 有效位置mask
        is_autoreg: 是否为自回归模型（选择最后一个元素）
    
    Returns:
        (modified_inputs, mfp_masks)
    """
    mfp_masks = get_initial_masks(input_columns, mask)
    
    # 选择单个元素
    selected_mask = select_single_element(mask, select_last=is_autoreg)
    
    modified_inputs = {}
    for key, column in input_columns.items():
        # ⭐ 跳过demo_only字段
        if column.get('demo_only', False):
            modified_inputs[key] = inputs.get(key, None)
            continue
        
        # ⭐ 跳过不存在的字段
        if key not in inputs:
            continue
        
        if not column.get('is_sequence', False):
            modified_inputs[key] = inputs[key]
        else:
            input_val = inputs[key]
            # ⭐ 跳过非tensor
            if not torch.is_tensor(input_val):
                modified_inputs[key] = input_val
                continue
            
            modified_inputs[key] = apply_token(input_val, column, selected_mask, 'masked')
            mfp_masks[key] = selected_mask
    
    return modified_inputs, mfp_masks


def feat_masking(
    inputs: Dict[str, torch.Tensor],
    input_columns: Dict,
    mask: torch.Tensor,
    feat_group: List[str]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    特征组masking（只mask特定的特征组）
    
    Args:
        inputs: 输入字典
        input_columns: 列配置
        mask: (B, S) 有效位置mask
        feat_group: 要mask的特征列表
    
    Returns:
        (modified_inputs, mfp_masks)
    """
    modified_inputs = {}
    for k, v in inputs.items():
        # ⭐ 只clone tensor类型
        if torch.is_tensor(v):
            modified_inputs[k] = v.clone()
        else:
            modified_inputs[k] = v
    
    mfp_masks = get_initial_masks(input_columns, mask)
    
    for key in feat_group:
        if key in input_columns and key in modified_inputs:
            column = input_columns[key]
            
            # ⭐ 跳过demo_only
            if column.get('demo_only', False):
                continue
            
            input_val = modified_inputs[key]
            # ⭐ 跳过非tensor
            if not torch.is_tensor(input_val):
                continue
            
            modified_inputs[key] = apply_token(input_val, column, mask, 'masked')
            mfp_masks[key] = mask
    
    return modified_inputs, mfp_masks


def select_single_element(mask: torch.Tensor, select_last: bool = False) -> torch.Tensor:
    """
    为每个样本选择单个元素
    
    Args:
        mask: (B, S) 有效位置mask
        select_last: 是否选择最后一个有效元素
    
    Returns:
        selected_mask: (B, S) 只有一个位置为True
    """
    batch_size, seq_len = mask.shape
    device = mask.device
    
    # 计算每个样本的有效长度
    length = mask.sum(dim=1).float()  # (B,)
    
    if select_last:
        # 选择最后一个有效位置
        indices = (length - 1).long()
    else:
        # 随机选择
        indices = (torch.rand(batch_size, device=device) * length).long()
    
    # 创建one-hot mask
    new_mask = F.one_hot(indices, num_classes=seq_len).bool()
    
    # 如果length=0，则全为False
    new_mask = new_mask & (length > 0).unsqueeze(1)
    
    return new_mask


def preprocess_for_train(
    inputs: Dict[str, torch.Tensor],
    input_columns: Dict,
    task_id: int,
    is_autoreg: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    训练时的预处理
    
    Args:
        inputs: 原始输入
        input_columns: 列配置
        task_id: 任务ID (0=random, 1=elem, 2+=feat groups)
        is_autoreg: 是否为自回归模型
    
    Returns:
        (targets, modified_inputs, masks)
    """
    # 生成序列mask
    seq_mask = get_seq_mask(inputs['length'], max_len=inputs['left'].size(1))
    
    # 过滤padding
    filtered_inputs = filter_padding(inputs, input_columns, seq_mask)
    
    # 获取属性组
    attribute_groups = get_attribute_groups(input_columns.keys())
    
    # 根据task_id选择masking策略
    if task_id == 0:
        # Random masking
        modified_inputs, masks = random_masking(filtered_inputs, input_columns, seq_mask)
    elif task_id == 1:
        # Element masking
        modified_inputs, masks = elem_masking(filtered_inputs, input_columns, seq_mask, is_autoreg)
    else:
        # Feature group masking
        group_names = list(attribute_groups.keys())
        group_idx = task_id - 2
        if group_idx < len(group_names):
            feat_group = attribute_groups[group_names[group_idx]]
            modified_inputs, masks = feat_masking(filtered_inputs, input_columns, seq_mask, feat_group)
        else:
            # 默认使用random
            modified_inputs, masks = random_masking(filtered_inputs, input_columns, seq_mask)
    
    return filtered_inputs, modified_inputs, masks


def preprocess_for_test(
    inputs: Dict[str, torch.Tensor],
    input_columns: Dict,
    masks: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    测试时的预处理（用于demo）
    
    Args:
        inputs: 原始输入
        input_columns: 列配置
        masks: 预定义的mask
    
    Returns:
        modified_inputs: 处理后的输入
    """
    seq_mask = get_seq_mask(inputs['length'], max_len=inputs['left'].size(1))
    filtered_inputs = filter_padding(inputs, input_columns, seq_mask)
    
    modified_inputs = {}
    for key, column in input_columns.items():
        if not column.get('is_sequence', False):
            modified_inputs[key] = filtered_inputs[key]
        else:
            modified_inputs[key] = apply_token(
                filtered_inputs[key], 
                column, 
                masks[key], 
                'masked'
            )
    
    return modified_inputs


def merge_inputs_and_prediction(
    inputs: Dict[str, torch.Tensor],
    input_columns: Dict,
    masks: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    合并输入和预测（未mask的部分保持原值）
    
    Args:
        inputs: 原始输入
        input_columns: 列配置
        masks: mask字典（True表示被mask的位置）
        predictions: 模型预测
    
    Returns:
        merged: 合并后的结果
    """
    merged = {}
    
    for key, column in input_columns.items():
        if key not in predictions:
            merged[key] = inputs[key]
            continue
        
        if not column.get('is_sequence', False):
            # 非序列特征保持原值
            merged[key] = inputs[key]
        elif column['type'] == 'numerical':
            # 数值特征
            pred = predictions[key]
            mask = masks[key]
            while mask.dim() < pred.dim():
                mask = mask.unsqueeze(-1)
            merged[key] = torch.where(mask, pred, inputs[key])
        else:
            # 分类特征（需要转换为one-hot）
            pred = predictions[key]  # (B, S, ..., C)
            gt = F.one_hot(inputs[key].long(), num_classes=column['input_dim'])
            
            mask = masks[key]
            while mask.dim() < gt.dim():
                mask = mask.unsqueeze(-1)
            
            merged[key] = torch.where(mask, pred, gt.float())
    
    # 复制demo_only字段
    for key, column in input_columns.items():
        if column.get('demo_only', False):
            merged[key] = inputs[key]
    
    return merged
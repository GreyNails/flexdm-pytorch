"""
PyTorch数据加载器
用于加载转换后的JSON格式设计数据
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional


class DesignLayoutDataset(Dataset):
    """设计布局数据集"""
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        max_length: int = 20,
        bins: int = 64,
    ):
        """
        Args:
            data_path: JSON数据文件路径
            split: 数据集划分 ('train', 'val', 'test')
            max_length: 最大元素数量
            bins: 位置离散化的区间数
        """
        self.data_path = Path(data_path)
        self.split = split
        self.max_length = max_length
        self.bins = bins
        
        # 加载数据
        json_file = self.data_path / f"{split}.json"
        print(f"加载数据: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"✓ 加载了 {len(self.data)} 个样本")
        
        # 加载词汇表
        vocab_file = self.data_path.parent / "vocabulary.json"
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocabulary = json.load(f)
        
        # 构建查找表
        self._build_lookups()
    
    def _build_lookups(self):
        """构建字符串到索引的映射"""
        self.type_to_idx = {v: i for i, v in enumerate(self.vocabulary['type'])}
        self.font_to_idx = {v: i for i, v in enumerate(self.vocabulary.get('font_family', []))}
        
        # 添加特殊token
        self.type_to_idx['<MASK>'] = len(self.type_to_idx)
        self.type_to_idx['<NULL>'] = len(self.type_to_idx)
        self.font_to_idx['<MASK>'] = len(self.font_to_idx)
        self.font_to_idx['<NULL>'] = len(self.font_to_idx)
    
    def discretize(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> int:
        """将连续值离散化到bins"""
        value = np.clip(value, min_val, max_val)
        return int((value - min_val) / (max_val - min_val) * (self.bins - 1))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        item = self.data[idx]
        length = min(item['length'], self.max_length)
        
        # 准备返回字典
        sample = {
            'id': item['id'],
            'length': torch.tensor([length], dtype=torch.long),
            'canvas_width': torch.tensor([item['canvas_width']], dtype=torch.long),
            'canvas_height': torch.tensor([item['canvas_height']], dtype=torch.long),
        }
        
        # 处理序列特征
        for key in ['left', 'top', 'width', 'height']:
            # 离散化位置和尺寸
            values = [self.discretize(item[key][i]) for i in range(length)]
            # Padding到max_length
            values += [0] * (self.max_length - length)
            sample[key] = torch.tensor(values, dtype=torch.long).unsqueeze(-1)
        
        # 类型编码
        type_ids = [self.type_to_idx.get(item['type'][i], 0) for i in range(length)]
        type_ids += [self.type_to_idx['<NULL>']] * (self.max_length - length)
        sample['type'] = torch.tensor(type_ids, dtype=torch.long).unsqueeze(-1)
        
        # 不透明度（归一化值）
        opacity = item['opacity'][:length] + [0.0] * (self.max_length - length)
        sample['opacity'] = torch.tensor(opacity, dtype=torch.float32).unsqueeze(-1)
        
        # 颜色 (RGB)
        colors = []
        for i in range(length):
            colors.append(item['color'][i])
        # Padding
        for _ in range(self.max_length - length):
            colors.append([0, 0, 0])
        sample['color'] = torch.tensor(colors, dtype=torch.long)
        
        # 字体
        if 'font_family' in item:
            font_ids = [self.font_to_idx.get(item['font_family'][i], 0) for i in range(length)]
            font_ids += [self.font_to_idx['<NULL>']] * (self.max_length - length)
            sample['font_family'] = torch.tensor(font_ids, dtype=torch.long).unsqueeze(-1)
        
        # 嵌入向量
        if 'image_embedding' in item:
            image_embs = item['image_embedding'][:length]
            # Padding
            for _ in range(self.max_length - length):
                image_embs.append([0.0] * 512)
            sample['image_embedding'] = torch.tensor(image_embs, dtype=torch.float32)
        
        if 'text_embedding' in item:
            text_embs = item['text_embedding'][:length]
            # Padding
            for _ in range(self.max_length - length):
                text_embs.append([0.0] * 512)
            sample['text_embedding'] = torch.tensor(text_embs, dtype=torch.float32)
        
        return sample


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """批处理函数"""
    # 获取所有键
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        if key == 'id':
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = torch.stack([item[key] for item in batch])
    
    return collated


def create_dataloader(
    data_path: str,
    split: str = 'train',
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **dataset_kwargs
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        data_path: 数据路径
        split: 数据集划分
        batch_size: 批大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        **dataset_kwargs: 传递给Dataset的额外参数
    
    Returns:
        DataLoader
    """
    dataset = DesignLayoutDataset(data_path, split=split, **dataset_kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return dataloader


# 测试代码
if __name__ == "__main__":
    # 测试数据加载
    data_path = "/home/dell/Project-HCL/BaseLine/flex-dm/data/crello_json"
    
    # 创建训练集加载器
    train_loader = create_dataloader(
        data_path=data_path,
        split='train',
        batch_size=16,
        shuffle=True,
    )
    
    print(f"训练集批次数: {len(train_loader)}")
    
    # 测试一个批次
    batch = next(iter(train_loader))
    print("\n样本批次:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:20s}: shape={list(value.shape)}, dtype={value.dtype}")
        else:
            print(f"  {key:20s}: {type(value)}")
    
    # 显示第一个样本
    print(f"\n第一个样本ID: {batch['id'][0]}")
    print(f"长度: {batch['length'][0].item()}")
    print(f"画布大小: {batch['canvas_width'][0].item()} x {batch['canvas_height'][0].item()}")
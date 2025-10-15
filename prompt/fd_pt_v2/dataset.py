"""
修复版 Dataset - 确保所有值都在正确范围内
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional


class DesignLayoutDataset(Dataset):
    """设计布局数据集 - 修复版"""
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        max_length: int = 20,
        bins: int = 64,
        min_font_freq: int = 500,
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.max_length = max_length
        self.bins = bins
        self.min_font_freq = min_font_freq
        
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
        print("\n构建查找表...")
        
        # === Type映射 - 关键修复：不包含特殊token ===
        # type_vocab = self.vocabulary['type']

        type_vocab = ['','svgElement', 'textElement', 'imageElement', 'coloredBackground', 'maskElement']


        # if isinstance(type_vocab, list):
        #     # 只映射实际的类型，索引从0开始
        #     self.type_to_idx = {v: i for i, v in enumerate(type_vocab)}
        # else:
        #     self.type_to_idx = {k: i for i, k in enumerate(type_vocab.keys())}
        
        self.type_to_idx={
            '':0,
            'svgElement': 1,
            'textElement': 2,
            'imageElement': 3,
            'coloredBackground': 4,
            'maskElement': 5
            }



        
        # 添加未知类型映射到0
        # self.type_to_idx['<UNKNOWN>'] = 0
        self.type_vocab_size = len(type_vocab)  # 不包含特殊token
        
        print(f"  Type词汇表: {self.type_vocab_size} 个类型")
        print(f"  Type映射: {self.type_to_idx}")
        
        # === Canvas Width映射 ===
        if 'canvas_width' in self.vocabulary:
            width_vocab = self.vocabulary['canvas_width']
            if isinstance(width_vocab, dict):
                widths = sorted([int(k) for k in width_vocab.keys()])
            elif isinstance(width_vocab, list):
                widths = sorted([int(v) for v in width_vocab])
            else:
                widths = list(range(200, 2001, 100))
            
            self.width_to_idx = {w: i for i, w in enumerate(widths)}
            self.idx_to_width = {i: w for i, w in enumerate(widths)}
            self.idx_to_width[-1] = widths[0] if widths else 800  # 默认值
            
            self.width_vocab_size = len(widths)
            print(f"  Canvas Width词汇表: {len(widths)} 个尺寸")
        else:
            self.width_to_idx = {}
            self.idx_to_width = {0: 800}
            self.width_vocab_size = 1
        
        # === Canvas Height映射 ===
        if 'canvas_height' in self.vocabulary:
            height_vocab = self.vocabulary['canvas_height']
            if isinstance(height_vocab, dict):
                heights = sorted([int(k) for k in height_vocab.keys()])
            elif isinstance(height_vocab, list):
                heights = sorted([int(v) for v in height_vocab])
            else:
                heights = list(range(200, 2001, 100))
            
            self.height_to_idx = {h: i for i, h in enumerate(heights)}
            self.idx_to_height = {i: h for i, h in enumerate(heights)}
            self.idx_to_height[-1] = heights[0] if heights else 600
            
            self.height_vocab_size = len(heights)
            print(f"  Canvas Height词汇表: {len(heights)} 个尺寸")
        else:
            self.height_to_idx = {}
            self.idx_to_height = {0: 600}
            self.height_vocab_size = 1
        
        # === Font映射 ===
        if 'font_family' in self.vocabulary:
            font_vocab = self.vocabulary['font_family']
            
            if isinstance(font_vocab, dict):
                filtered_fonts = [
                    font for font, count in font_vocab.items() 
                    if count >= self.min_font_freq
                ]
                filtered_fonts.sort()
                self.font_to_idx = {font: i for i, font in enumerate(filtered_fonts)}
            else:
                self.font_to_idx = {v: i for i, v in enumerate(font_vocab)}
            
            # 关键修复：OOV索引应该是0（未知），而不是超出范围的值
            self.font_vocab_size = len(self.font_to_idx)
            # OOV映射到0
            self.font_oov_idx = 0
            
            print(f"  Font词汇表: {self.font_vocab_size} 个字体 (OOV->0)")
        else:
            self.font_to_idx = {}
            self.font_oov_idx = 0
            self.font_vocab_size = 0
        
        # 反向映射
        self.idx_to_type = {v: k for k, v in self.type_to_idx.items()}
        self.idx_to_font = {v: k for k, v in self.font_to_idx.items()}
    
    def discretize(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> int:
        """将连续值离散化到bins，确保结果在 [0, bins-1] 范围内"""
        value = np.clip(value, min_val, max_val)
        discrete = int((value - min_val) / (max_val - min_val) * (self.bins - 1))
        return np.clip(discrete, 0, self.bins - 1)  # 确保不超出范围
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """获取单个样本 - 修复版，确保所有值都在范围内"""
        item = self.data[idx]
        length = min(item['length'], self.max_length)
        
        # Canvas尺寸
        canvas_w = item['canvas_width']
        canvas_h = item['canvas_height']
        
        width_idx = self.width_to_idx.get(canvas_w, 0)
        height_idx = self.height_to_idx.get(canvas_h, 0)
        
        if width_idx == 0 and canvas_w not in self.width_to_idx:
            closest_w = min(self.width_to_idx.keys(), 
                          key=lambda x: abs(x - canvas_w)) if self.width_to_idx else 800
            width_idx = self.width_to_idx.get(closest_w, 0)
            
        if height_idx == 0 and canvas_h not in self.height_to_idx:
            closest_h = min(self.height_to_idx.keys(), 
                          key=lambda x: abs(x - canvas_h)) if self.height_to_idx else 600
            height_idx = self.height_to_idx.get(closest_h, 0)
        
        sample = {
            'id': item['id'],
            'length': torch.tensor([length], dtype=torch.long),
            'canvas_width': torch.tensor([width_idx], dtype=torch.long),
            'canvas_height': torch.tensor([height_idx], dtype=torch.long),
        }
        
        # 位置和尺寸 - 确保在 [0, bins-1] 范围内
        for key in ['left', 'top', 'width', 'height']:
            values = [self.discretize(item[key][i]) for i in range(length)]
            values += [0] * (self.max_length - length)
            sample[key] = torch.tensor(values, dtype=torch.long).unsqueeze(-1)
        
        # 类型编码 - 确保在 [0, type_vocab_size-1] 范围内
        type_ids = []
        for i in range(length):
            type_name = item['type'][i]
            type_id = self.type_to_idx.get(type_name, 0)  # 未知类型映射到0
            type_id = min(type_id, self.type_vocab_size - 1)  # 确保不超出范围
            type_ids.append(type_id)
        type_ids += [0] * (self.max_length - length)
        sample['type'] = torch.tensor(type_ids, dtype=torch.long).unsqueeze(-1)
        
        # 不透明度 - 确保在 [0, 7] 范围内
        if 'opacity' in item:
            opacity_values = []
            for i in range(length):
                # 离散化到8个bins: 0.0-1.0 -> 0-7
                opacity = np.clip(item['opacity'][i], 0.0, 1.0)
                discrete_val = int(opacity * 7)
                discrete_val = min(discrete_val, 7)  # 确保不超过7
                opacity_values.append(discrete_val)
            opacity_values += [0] * (self.max_length - length)
            sample['opacity'] = torch.tensor(opacity_values, dtype=torch.long).unsqueeze(-1)
        
        # 颜色 - 确保每个通道在 [0, 15] 范围内
        if 'color' in item:
            colors = []
            for i in range(length):
                rgb = item['color'][i]
                # 离散化每个通道：0-255 -> 0-15
                discrete_rgb = []
                for c in rgb:
                    c = np.clip(c, 0, 255)
                    discrete_c = int(c * 15 / 255)
                    discrete_c = min(discrete_c, 15)  # 确保不超过15
                    discrete_rgb.append(discrete_c)
                colors.append(discrete_rgb)
            for _ in range(self.max_length - length):
                colors.append([0, 0, 0])
            sample['color'] = torch.tensor(colors, dtype=torch.long)
        
        # 字体编码 - 确保在 [0, font_vocab_size-1] 范围内
        if 'font_family' in item and self.font_to_idx:
            font_ids = []
            for i in range(length):
                font_name = item['font_family'][i]
                font_id = self.font_to_idx.get(font_name, self.font_oov_idx)
                # 关键修复：确保不超出范围 [0, font_vocab_size-1]
                font_id = np.clip(font_id, 0, self.font_vocab_size - 1)
                font_ids.append(font_id)
            
            font_ids += [0] * (self.max_length - length)
            sample['font_family'] = torch.tensor(font_ids, dtype=torch.long).unsqueeze(-1)
        
        # UUID - 仅用于demo，不参与训练
        if 'uuid' in item:
            # 简单存储原始值，但标记为demo_only
            uuid_vals = item['uuid'][:length] + [''] * (self.max_length - length)
            sample['uuid'] = uuid_vals  # 保持为字符串列表，不转tensor
        
        # 图像嵌入
        if 'image_embedding' in item:
            image_embs = item['image_embedding'][:length]
            for _ in range(self.max_length - length):
                image_embs.append([0.0] * 512)
            sample['image_embedding'] = torch.tensor(image_embs, dtype=torch.float32)
        
        # 文本嵌入
        if 'text_embedding' in item:
            text_embs = item['text_embedding'][:length]
            for _ in range(self.max_length - length):
                text_embs.append([0.0] * 512)
            sample['text_embedding'] = torch.tensor(text_embs, dtype=torch.float32)
        
        return sample
    
    def get_input_columns(self) -> Dict:
        """
        生成input_columns配置
        关键：input_dim 是实际的类别数，不包含Encoder会添加的特殊token
        """
        input_columns = {
            'id': {
                'demo_only': True,
                'shape': [1],
                'is_sequence': False,
                'primary_label': None,
            },
            'length': {
                'type': 'categorical',
                'input_dim': 50,
                'shape': [1],
                'is_sequence': False,
                'primary_label': None,
            },
            'canvas_width': {
                'type': 'categorical',
                'input_dim': self.width_vocab_size,
                'shape': [1],
                'is_sequence': False,
                'primary_label': None,
            },
            'canvas_height': {
                'type': 'categorical',
                'input_dim': self.height_vocab_size,
                'shape': [1],
                'is_sequence': False,
                'primary_label': None,
            },
            'type': {
                'type': 'categorical',
                'input_dim': self.type_vocab_size,  # 实际类别数
                'shape': [1],
                'is_sequence': True,
                'primary_label': 0,
            },
            'left': {
                'type': 'categorical',
                'input_dim': self.bins,  # 64
                'shape': [1],
                'is_sequence': True,
                'primary_label': None,
            },
            'top': {
                'type': 'categorical',
                'input_dim': self.bins,  # 64
                'shape': [1],
                'is_sequence': True,
                'primary_label': None,
            },
            'width': {
                'type': 'categorical',
                'input_dim': self.bins,  # 64
                'shape': [1],
                'is_sequence': True,
                'primary_label': None,
            },
            'height': {
                'type': 'categorical',
                'input_dim': self.bins,  # 64
                'shape': [1],
                'is_sequence': True,
                'primary_label': None,
            },
            'opacity': {
                'type': 'categorical',
                'input_dim': 8,  # 0-7
                'shape': [1],
                'is_sequence': True,
                'primary_label': None,
            },
            'color': {
                'type': 'categorical',
                'input_dim': 16,  # 0-15 每个通道
                'shape': [3],
                'is_sequence': True,
                'primary_label': None,
            },
            'image_embedding': {
                'type': 'numerical',
                'shape': [512],
                'is_sequence': True,
                'primary_label': None,
            },
            'text_embedding': {
                'type': 'numerical',
                'shape': [512],
                'is_sequence': True,
                'primary_label': None,
            },
        }
        
        # 只有在有字体数据时才添加
        if self.font_vocab_size > 0:
            input_columns['font_family'] = {
                'type': 'categorical',
                'input_dim': self.font_vocab_size,
                'shape': [1],
                'is_sequence': True,
                'primary_label': None,
            }
        
        # UUID 仅用于演示，不参与训练
        input_columns['uuid'] = {
            'demo_only': True,
            'type': 'categorical',
            'input_dim': 1215,
            'shape': [1],
            'is_sequence': True,
            'primary_label': None,
        }
        
        return input_columns


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """批处理函数"""
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        if key in ['id', 'uuid']:  # id和uuid保持为列表
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
    """创建数据加载器"""
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
"""
PyTorch数据加载器 - 修复版
生成与TF版本一致的input_columns格式
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
        
        # === Type映射 ===
        type_vocab = self.vocabulary['type']
        if isinstance(type_vocab, list):
            self.type_to_idx = {v: i+1 for i, v in enumerate(type_vocab)}
        else:
            self.type_to_idx = {k: i+1 for i, k in enumerate(type_vocab.keys())}
        
        type_vocab_size = len(self.type_to_idx)
        self.type_to_idx['<NULL>'] = 0
        self.type_to_idx['<MASK>'] = type_vocab_size + 1
        
        print(f"  Type词汇表: {len(self.type_to_idx)} 个类型")
        
        # === Canvas Width映射 ===
        if 'canvas_width' in self.vocabulary:
            width_vocab = self.vocabulary['canvas_width']
            if isinstance(width_vocab, dict):
                widths = sorted([int(k) for k in width_vocab.keys()])
            elif isinstance(width_vocab, list):
                widths = sorted([int(v) for v in width_vocab])
            else:
                widths = list(range(200, 2001, 100))
            
            self.width_to_idx = {w: i+1 for i, w in enumerate(widths)}
            self.idx_to_width = {i+1: w for i, w in enumerate(widths)}
            self.idx_to_width[0] = widths[0] if widths else 800
            
            self.width_vocab_size = len(widths) + 1
            print(f"  Canvas Width词汇表: {len(widths)} 个尺寸")
            print(f"    范围: {min(widths)} - {max(widths)}")
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
            
            self.height_to_idx = {h: i+1 for i, h in enumerate(heights)}
            self.idx_to_height = {i+1: h for i, h in enumerate(heights)}
            self.idx_to_height[0] = heights[0] if heights else 600
            
            self.height_vocab_size = len(heights) + 1
            print(f"  Canvas Height词汇表: {len(heights)} 个尺寸")
            print(f"    范围: {min(heights)} - {max(heights)}")
        else:
            self.height_to_idx = {}
            self.idx_to_height = {0: 600}
            self.height_vocab_size = 1
        
        # === Font映射 ===
        if 'font_family' in self.vocabulary:
            font_vocab = self.vocabulary['font_family']
            
            if isinstance(font_vocab, dict):
                total_fonts = len(font_vocab)
                filtered_fonts = [
                    font for font, count in font_vocab.items() 
                    if count >= self.min_font_freq
                ]
                filtered_fonts.sort()
                
                print(f"  Font过滤: {total_fonts} -> {len(filtered_fonts)} (频率>={self.min_font_freq})")
                self.font_to_idx = {font: i+1 for i, font in enumerate(filtered_fonts)}
            else:
                self.font_to_idx = {v: i+1 for i, v in enumerate(font_vocab)}
            
            vocab_size = len(self.font_to_idx)
            self.font_to_idx['<NULL>'] = 0
            self.font_to_idx['<OOV>'] = vocab_size + 1
            self.font_to_idx['<MASK>'] = vocab_size + 2
            
            self.font_oov_idx = vocab_size + 1
            self.font_vocab_size = vocab_size + 2
            
            print(f"  Font词汇表: {len(self.font_to_idx)} 个token (含特殊token)")
        else:
            self.font_to_idx = {}
            self.font_oov_idx = 0
            self.font_vocab_size = 0
        
        # 反向映射
        self.idx_to_type = {v: k for k, v in self.type_to_idx.items()}
        self.idx_to_font = {v: k for k, v in self.font_to_idx.items()}
    
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
        
        # Canvas尺寸
        canvas_w = item['canvas_width']
        canvas_h = item['canvas_height']
        
        width_idx = self.width_to_idx.get(canvas_w, 0)
        height_idx = self.height_to_idx.get(canvas_h, 0)
        
        if width_idx == 0 and self.width_to_idx:
            closest_w = min(self.width_to_idx.keys(), key=lambda x: abs(x - canvas_w))
            width_idx = self.width_to_idx[closest_w]
            
        if height_idx == 0 and self.height_to_idx:
            closest_h = min(self.height_to_idx.keys(), key=lambda x: abs(x - canvas_h))
            height_idx = self.height_to_idx[closest_h]
        
        sample = {
            'id': item['id'],
            'length': torch.tensor([length], dtype=torch.long),
            'canvas_width': torch.tensor([width_idx], dtype=torch.long),
            'canvas_height': torch.tensor([height_idx], dtype=torch.long),
        }
        
        # 位置和尺寸
        for key in ['left', 'top', 'width', 'height']:
            values = [self.discretize(item[key][i]) for i in range(length)]
            values += [0] * (self.max_length - length)
            sample[key] = torch.tensor(values, dtype=torch.long).unsqueeze(-1)
        
        # 类型编码
        type_ids = [self.type_to_idx.get(item['type'][i], 0) for i in range(length)]
        type_ids += [0] * (self.max_length - length)
        sample['type'] = torch.tensor(type_ids, dtype=torch.long).unsqueeze(-1)
        
        # 不透明度
        if 'opacity' in item:
            opacity = item['opacity'][:length] + [0.0] * (self.max_length - length)
            sample['opacity'] = torch.tensor(opacity, dtype=torch.float32).unsqueeze(-1)
        
        # 颜色
        if 'color' in item:
            colors = []
            for i in range(length):
                colors.append(item['color'][i])
            for _ in range(self.max_length - length):
                colors.append([0, 0, 0])
            sample['color'] = torch.tensor(colors, dtype=torch.long)
        
        # 字体编码
        if 'font_family' in item and self.font_to_idx:
            font_ids = []
            for i in range(length):
                font_name = item['font_family'][i]
                font_id = self.font_to_idx.get(font_name, self.font_oov_idx)
                font_ids.append(font_id)
            
            font_ids += [0] * (self.max_length - length)
            sample['font_family'] = torch.tensor(font_ids, dtype=torch.long).unsqueeze(-1)
        
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
        
        # UUID (demo only)
        if 'uuid' in item:
            uuid_vals = item['uuid'][:length] + [''] * (self.max_length - length)
            # 简单哈希uuid到整数
            uuid_ids = [hash(u) % 10000 for u in uuid_vals]
            sample['uuid'] = torch.tensor(uuid_ids, dtype=torch.long).unsqueeze(-1)
        
        return sample
    
    def get_input_columns(self) -> Dict:
        """
        生成input_columns配置 - 严格对齐TF格式
        
        关键修复:
        1. opacity的input_dim应该是8(离散化的bins数)
        2. color的shape是[3]，input_dim是16
        3. uuid标记为demo_only
        4. 添加loss_condition字段
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
                'input_dim': 50,  # 原始设置
                'shape': [1],
                'is_sequence': False,
                'primary_label': None,
            },
            'canvas_width': {
                'type': 'categorical',
                'input_dim': self.width_vocab_size - 1,  # 减去padding
                'shape': [1],
                'is_sequence': False,
                'primary_label': None,
            },
            'canvas_height': {
                'type': 'categorical',
                'input_dim': self.height_vocab_size - 1,
                'shape': [1],
                'is_sequence': False,
                'primary_label': None,
            },
            'type': {
                'type': 'categorical',
                'input_dim': len(self.type_to_idx) - 2,  # 不包含<NULL>和<MASK>
                'shape': [1],
                'is_sequence': True,
                'primary_label': 0,
            },
            'left': {
                'type': 'categorical',
                'input_dim': self.bins,
                'shape': [1],
                'is_sequence': True,
                'primary_label': None,
            },
            'top': {
                'type': 'categorical',
                'input_dim': self.bins,
                'shape': [1],
                'is_sequence': True,
                'primary_label': None,
            },
            'width': {
                'type': 'categorical',
                'input_dim': self.bins,
                'shape': [1],
                'is_sequence': True,
                'primary_label': None,
            },
            'height': {
                'type': 'categorical',
                'input_dim': self.bins,
                'shape': [1],
                'is_sequence': True,
                'primary_label': None,
            },
        }
        
        # Opacity - 🔧 修复：应该与bins保持一致
        if any('opacity' in item for item in self.data[:10]):
            input_columns['opacity'] = {
                'type': 'categorical',
                'input_dim': self.bins,  # 使用实际的bins数量(64)
                'shape': [1],
                'is_sequence': True,
                'primary_label': None,
            }
        
        # Color - 关键修复：shape=[3], input_dim=16
        if any('color' in item for item in self.data[:10]):
            type_idx = list(self.type_to_idx.keys())
            input_columns['color'] = {
                'type': 'categorical',
                'input_dim': 16,  # 固定为16
                'shape': [3],  # RGB三通道
                'is_sequence': True,
                'primary_label': None,
                'loss_condition': {
                    'key': 'type',
                    'mask': [
                        False,  # NULL
                        False,  # svgElement
                        True,   # textElement
                        False,  # imageElement
                        True,   # coloredBackground
                        False,  # maskElement
                    ]
                }
            }
        
        # Image embedding
        if any('image_embedding' in item for item in self.data[:10]):
            input_columns['image_embedding'] = {
                'type': 'numerical',
                'shape': [512],
                'is_sequence': True,
                'primary_label': None,
                'loss_condition': {
                    'key': 'type',
                    'mask': [
                        False,  # NULL
                        True,   # svgElement
                        False,  # textElement
                        True,   # imageElement
                        False,  # coloredBackground
                        True,   # maskElement
                    ]
                }
            }
        
        # Text embedding
        if any('text_embedding' in item for item in self.data[:10]):
            input_columns['text_embedding'] = {
                'type': 'numerical',
                'shape': [512],
                'is_sequence': True,
                'primary_label': None,
                'loss_condition': {
                    'key': 'type',
                    'mask': [
                        False,  # NULL
                        False,  # svgElement
                        True,   # textElement
                        False,  # imageElement
                        False,  # coloredBackground
                        False,  # maskElement
                    ]
                }
            }
        
        # Font family
        if self.font_to_idx:
            input_columns['font_family'] = {
                'type': 'categorical',
                'input_dim': self.font_vocab_size - 2,  # 不包含<NULL>和<MASK>
                'shape': [1],
                'is_sequence': True,
                'primary_label': None,
                'loss_condition': {
                    'key': 'type',
                    'mask': [
                        False,  # NULL
                        False,  # svgElement
                        True,   # textElement
                        False,  # imageElement
                        False,  # coloredBackground
                        False,  # maskElement
                    ]
                }
            }
        
        # UUID - demo only
        if any('uuid' in item for item in self.data[:10]):
            input_columns['uuid'] = {
                'demo_only': True,
                'type': 'categorical',
                'input_dim': 10000,
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


if __name__ == "__main__":
    data_path = "/home/dell/Project-HCL/BaseLine/flex-dm/data/crello_json"
    
    print("="*60)
    print("数据集测试")
    print("="*60)
    
    train_dataset = DesignLayoutDataset(
        data_path=data_path,
        split='train',
        max_length=20,
        min_font_freq=500,
    )
    
    train_loader = create_dataloader(
        data_path=data_path,
        split='train',
        batch_size=16,
        shuffle=True,
    )
    
    print(f"\n训练集批次数: {len(train_loader)}")
    
    batch = next(iter(train_loader))
    print("\n样本批次:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:20s}: shape={list(value.shape)}, dtype={value.dtype}")
        else:
            print(f"  {key:20s}: {type(value)}")
    
    input_columns = train_dataset.get_input_columns()
    print(f"\n生成的input_columns:")
    for key, config in input_columns.items():
        print(f"  {key:20s}: {config}")
    
    import json
    output_file = "input_columns_fixed.json"
    with open(output_file, 'w') as f:
        json.dump(input_columns, f, indent=2)
    print(f"\n✓ 配置已保存到: {output_file}")
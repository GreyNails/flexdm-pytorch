"""
PyTorch数据加载器
用于加载转换后的JSON格式设计数据
修复了font_family编码问题
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
        min_font_freq: int = 500,  # 字体最小频率阈值
    ):
        """
        Args:
            data_path: JSON数据文件路径
            split: 数据集划分 ('train', 'val', 'test')
            max_length: 最大元素数量
            bins: 位置离散化的区间数
            min_font_freq: 字体最小出现频率
        """
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
        """构建字符串到索引的映射（修复版）"""
        print("\n构建查找表...")
        
        # === Type映射 ===
        type_vocab = self.vocabulary['type']
        if isinstance(type_vocab, list):
            self.type_to_idx = {v: i+1 for i, v in enumerate(type_vocab)}  # 从1开始
        else:
            # 如果是字典格式
            self.type_to_idx = {k: i+1 for i, k in enumerate(type_vocab.keys())}
        
        # 添加特殊token（0保留给padding）
        type_vocab_size = len(self.type_to_idx)
        self.type_to_idx['<NULL>'] = 0  # padding
        self.type_to_idx['<MASK>'] = type_vocab_size + 1
        
        print(f"  Type词汇表: {len(self.type_to_idx)} 个类型")
        
        # === Font映射（关键修复） ===
        if 'font_family' in self.vocabulary:
            font_vocab = self.vocabulary['font_family']
            
            if isinstance(font_vocab, dict):
                # 字典格式：{"font_name": count}
                total_fonts = len(font_vocab)
                
                # 频率过滤
                filtered_fonts = [
                    font for font, count in font_vocab.items() 
                    if count >= self.min_font_freq
                ]
                filtered_fonts.sort()  # 排序保证一致性
                
                print(f"  Font过滤: {total_fonts} -> {len(filtered_fonts)} (频率>={self.min_font_freq})")
                
                # 构建映射（从1开始，0留给padding）
                self.font_to_idx = {font: i+1 for i, font in enumerate(filtered_fonts)}
            else:
                # 列表格式
                self.font_to_idx = {v: i+1 for i, v in enumerate(font_vocab)}
                print(f"  Font词汇表: {len(self.font_to_idx)} 个字体")
            
            # 添加特殊token
            vocab_size = len(self.font_to_idx)
            self.font_to_idx['<NULL>'] = 0           # padding
            self.font_to_idx['<OOV>'] = vocab_size + 1   # 未知字体
            self.font_to_idx['<MASK>'] = vocab_size + 2  # 训练时遮蔽
            
            self.font_oov_idx = vocab_size + 1
            self.font_vocab_size = vocab_size + 2  # 不包括padding的0
            
            print(f"  Font词汇表: {len(self.font_to_idx)} 个token (含特殊token)")
            print(f"    - 有效字体: {vocab_size}")
            print(f"    - OOV索引: {self.font_oov_idx}")
            print(f"    - MASK索引: {vocab_size + 2}")
        else:
            self.font_to_idx = {}
            self.font_oov_idx = 0
            self.font_vocab_size = 0
            print("  未找到font_family字段")
        
        # 反向映射（用于调试）
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
        
        # 准备返回字典
        sample = {
            'id': item['id'],
            'length': torch.tensor([length], dtype=torch.long),
            'canvas_width': torch.tensor([item['canvas_width']], dtype=torch.long),
            'canvas_height': torch.tensor([item['canvas_height']], dtype=torch.long),
        }
        
        # 处理序列特征 - 位置和尺寸
        for key in ['left', 'top', 'width', 'height']:
            values = [self.discretize(item[key][i]) for i in range(length)]
            values += [0] * (self.max_length - length)  # padding
            sample[key] = torch.tensor(values, dtype=torch.long).unsqueeze(-1)
        
        # 类型编码
        type_ids = [self.type_to_idx.get(item['type'][i], 0) for i in range(length)]
        type_ids += [0] * (self.max_length - length)  # padding用0
        sample['type'] = torch.tensor(type_ids, dtype=torch.long).unsqueeze(-1)
        
        # 不透明度（连续值）
        if 'opacity' in item:
            opacity = item['opacity'][:length] + [0.0] * (self.max_length - length)
            sample['opacity'] = torch.tensor(opacity, dtype=torch.float32).unsqueeze(-1)
        
        # 颜色 (RGB)
        if 'color' in item:
            colors = []
            for i in range(length):
                colors.append(item['color'][i])
            for _ in range(self.max_length - length):
                colors.append([0, 0, 0])  # padding
            sample['color'] = torch.tensor(colors, dtype=torch.long)
        
        # 字体编码（修复版）
        if 'font_family' in item and self.font_to_idx:
            font_ids = []
            for i in range(length):
                font_name = item['font_family'][i]
                # 使用OOV索引处理未知字体
                font_id = self.font_to_idx.get(font_name, self.font_oov_idx)
                font_ids.append(font_id)
            
            # Padding用0
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
        
        return sample
    
    def get_input_columns(self) -> Dict:
        """生成input_columns配置"""
        input_columns = {
            'type': {
                'is_sequence': True,
                'type': 'categorical',
                'input_dim': len(self.type_to_idx) - 1,  # 不包括<NULL>
                'shape': [1]
            },
            'left': {
                'is_sequence': True,
                'type': 'categorical',
                'input_dim': self.bins,
                'shape': [1]
            },
            'top': {
                'is_sequence': True,
                'type': 'categorical',
                'input_dim': self.bins,
                'shape': [1]
            },
            'width': {
                'is_sequence': True,
                'type': 'categorical',
                'input_dim': self.bins,
                'shape': [1]
            },
            'height': {
                'is_sequence': True,
                'type': 'categorical',
                'input_dim': self.bins,
                'shape': [1]
            },
        }
        
        # 字体
        if self.font_to_idx:
            input_columns['font_family'] = {
                'is_sequence': True,
                'type': 'categorical',
                'input_dim': self.font_vocab_size,
                'shape': [1],
                'loss_condition': {
                    'key': 'type',
                    'values': ['textElement']  # 只对文本元素计算损失
                }
            }
        
        # 不透明度
        if any('opacity' in item for item in self.data[:10]):
            input_columns['opacity'] = {
                'is_sequence': True,
                'type': 'categorical',  # 或 'numerical'
                'input_dim': 8,  # 8个bins
                'shape': [1]
            }
        
        # 颜色
        if any('color' in item for item in self.data[:10]):
            input_columns['color'] = {
                'is_sequence': True,
                'type': 'categorical',
                'input_dim': 16,  # 16个bins (per channel)
                'shape': [3]  # RGB
            }
        
        # 嵌入
        if any('image_embedding' in item for item in self.data[:10]):
            input_columns['image_embedding'] = {
                'is_sequence': True,
                'type': 'numerical',
                'shape': [512]
            }
        
        if any('text_embedding' in item for item in self.data[:10]):
            input_columns['text_embedding'] = {
                'is_sequence': True,
                'type': 'numerical',
                'shape': [512]
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


# ==================== 测试和验证代码 ====================

def validate_font_encoding(dataset: DesignLayoutDataset, num_samples: int = 5):
    """验证字体编码正确性"""
    print("\n" + "="*60)
    print("字体编码验证")
    print("="*60)
    
    print(f"\n1. 词汇表统计:")
    print(f"   总token数: {len(dataset.font_to_idx)}")
    print(f"   有效字体数: {dataset.font_vocab_size - 2}")
    print(f"   OOV索引: {dataset.font_oov_idx}")
    
    print(f"\n2. 前10个字体:")
    for i, (font, idx) in enumerate(list(dataset.font_to_idx.items())[:10]):
        print(f"   {idx:3d}: {font}")
    
    print(f"\n3. 特殊token:")
    for token in ['<NULL>', '<OOV>', '<MASK>']:
        if token in dataset.font_to_idx:
            print(f"   {token:8s}: {dataset.font_to_idx[token]}")
    
    print(f"\n4. 样本验证:")
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        if 'font_family' in sample:
            font_ids = sample['font_family'].squeeze().tolist()
            length = sample['length'].item()
            
            print(f"\n   样本 {i}:")
            print(f"   长度: {length}")
            print(f"   字体ID (前5个): {font_ids[:5]}")
            
            # 解码回字体名
            fonts_decoded = []
            for fid in font_ids[:length]:
                font_name = dataset.idx_to_font.get(fid, '<UNKNOWN>')
                fonts_decoded.append(font_name)
            print(f"   字体名 (前3个): {fonts_decoded[:3]}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # 测试数据加载
    data_path = "/home/dell/Project-HCL/BaseLine/flex-dm/data/crello_json"
    
    print("="*60)
    print("数据集测试")
    print("="*60)
    
    # 创建训练集
    train_dataset = DesignLayoutDataset(
        data_path=data_path,
        split='train',
        max_length=20,
        min_font_freq=500,
    )
    
    # 验证字体编码
    validate_font_encoding(train_dataset)
    
    # 创建DataLoader
    train_loader = create_dataloader(
        data_path=data_path,
        split='train',
        batch_size=16,
        shuffle=True,
    )
    
    print(f"\n训练集批次数: {len(train_loader)}")
    
    # 测试一个批次
    batch = next(iter(train_loader))
    print("\n样本批次:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:20s}: shape={list(value.shape)}, dtype={value.dtype}")
        else:
            print(f"  {key:20s}: {type(value)}")
    
    # 显示第一个样本
    print(f"\n第一个样本:")
    print(f"  ID: {batch['id'][0]}")
    print(f"  长度: {batch['length'][0].item()}")
    print(f"  画布: {batch['canvas_width'][0].item()} x {batch['canvas_height'][0].item()}")
    
    if 'font_family' in batch:
        print(f"  字体ID: {batch['font_family'][0, :5].squeeze().tolist()}")
    
    # 生成input_columns配置
    input_columns = train_dataset.get_input_columns()
    print(f"\n生成的input_columns:")
    for key, config in input_columns.items():
        print(f"  {key:20s}: {config}")
    
    # 保存配置
    import json
    output_file = "input_columns_generated.json"
    with open(output_file, 'w') as f:
        json.dump(input_columns, f, indent=2)
    print(f"\n✓ 配置已保存到: {output_file}")
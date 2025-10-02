"""
PyTorch模型架构 - 完全修复版本
使用ModuleList替代ModuleDict，避免键冲突
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math


# ==================== Transformer Components ====================

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        lookahead: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.lookahead = lookahead
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, S, D = x.shape
        
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float('-inf'))
            
            if not self.lookahead:
                causal_mask = torch.triu(
                    torch.ones(S, S, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
                scores = scores.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)
        return out


class TransformerBlock(nn.Module):
    """Transformer块（DeepSVG风格）"""
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        lookahead: bool = True,
    ):
        super().__init__()
        ff_dim = ff_dim or (2 * embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim, num_heads, dropout, lookahead
        )
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, mask)
        x = self.dropout1(x)
        x = residual + x
        
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = residual + x
        
        return x


class TransformerBlocks(nn.Module):
    """堆叠的Transformer块"""
    
    def __init__(
        self,
        num_blocks: int = 4,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        lookahead: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout, lookahead=lookahead)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        for block in self.blocks:
            x = block(x, mask)
        return x


# ==================== Encoder（修复版本）====================

class Encoder(nn.Module):
    """编码器 - 使用ModuleList避免键冲突"""
    
    def __init__(
        self,
        input_columns: Dict,
        embed_dim: int = 128,
        dropout: float = 0.1,
        max_length: int = 50,
    ):
        super().__init__()
        self.input_columns = input_columns
        self.embed_dim = embed_dim
        
        # 使用列表而不是字典存储层
        self.emb_layers = nn.ModuleList()
        self.emb_keys = []
        self.emb_types = []
        self.emb_configs = []
        
        print("初始化Encoder:")
        for key, column in input_columns.items():
            if not column.get('is_sequence', False):
                continue
            
            self.emb_keys.append(key)
            self.emb_types.append(column['type'])
            self.emb_configs.append(column)
            
            if column['type'] == 'categorical':
                vocab_size = column['input_dim'] + 2
                self.emb_layers.append(nn.Embedding(vocab_size, embed_dim))
                print(f"  {key}: Embedding({vocab_size}, {embed_dim})")
            elif column['type'] == 'numerical':
                input_size = column['shape'][-1] if 'shape' in column else 1
                self.emb_layers.append(nn.Linear(input_size, embed_dim))
                print(f"  {key}: Linear({input_size}, {embed_dim})")
        
        print(f"总计: {len(self.emb_keys)} 个特征")
        
        self.pos_embedding = nn.Embedding(max_length + 1, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> tuple:
        batch_size = inputs['length'].size(0)
        
        # 找到序列长度
        seq_len = None
        for key in self.emb_keys:
            if key in inputs:
                seq_len = inputs[key].size(1)
                break
        
        if seq_len is None:
            raise ValueError("未找到序列特征")
        
        # 编码每个特征
        seq_embs = []
        for idx, key in enumerate(self.emb_keys):
            if key not in inputs:
                continue
            
            x = inputs[key]
            emb = self.emb_layers[idx](x)
            
            # 处理多维特征（如RGB）
            if len(emb.shape) == 4:
                emb = emb.sum(dim=2)
            
            seq_embs.append(emb)
        
        # 融合特征
        seq = torch.stack(seq_embs).sum(dim=0)
        
        # 位置编码
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        seq = seq + self.pos_embedding(positions)
        seq = self.dropout(seq)
        
        # 生成掩码
        lengths = inputs['length'].squeeze(-1)
        mask = torch.arange(seq_len, device=seq.device).unsqueeze(0) < lengths.unsqueeze(1)
        
        return seq, mask


# ==================== Decoder（修复版本）====================

class Decoder(nn.Module):
    """解码器 - 使用ModuleList避免键冲突"""
    
    def __init__(self, input_columns: Dict, embed_dim: int = 128):
        super().__init__()
        self.input_columns = input_columns
        
        # 使用列表而不是字典存储层
        self.head_layers = nn.ModuleList()
        self.head_keys = []
        self.head_configs = []
        
        print("初始化Decoder:")
        for key, column in input_columns.items():
            if not column.get('is_sequence', False):
                continue
            
            self.head_keys.append(key)
            self.head_configs.append(column)
            
            if column['type'] == 'categorical':
                shape = column.get('shape', [1])
                output_dim = shape[-1] * column['input_dim']
                self.head_layers.append(nn.Linear(embed_dim, output_dim))
                print(f"  {key}: Linear({embed_dim}, {output_dim}) -> ({shape[-1]}, {column['input_dim']})")
            else:
                shape = column.get('shape', [1])
                output_dim = shape[-1]
                self.head_layers.append(nn.Linear(embed_dim, output_dim))
                print(f"  {key}: Linear({embed_dim}, {output_dim})")
        
        print(f"总计: {len(self.head_keys)} 个输出头")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}
        batch_size, seq_len, _ = x.shape
        
        for idx, key in enumerate(self.head_keys):
            column = self.head_configs[idx]
            pred = self.head_layers[idx](x)
            
            if column['type'] == 'categorical':
                shape = column.get('shape', [1])
                num_features = shape[-1]
                vocab_size = column['input_dim']
                pred = pred.view(batch_size, seq_len, num_features, vocab_size)
            
            outputs[key] = pred
        
        return outputs


# ==================== MFP Model ====================

class MFP(nn.Module):
    """Masked Field Prediction模型"""
    
    def __init__(
        self,
        input_columns: Dict,
        embed_dim: int = 128,
        num_blocks: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_length: int = 50,
    ):
        super().__init__()
        self.input_columns = input_columns
        
        print("\n" + "="*60)
        print("初始化MFP模型")
        print("="*60)
        
        self.encoder = Encoder(
            input_columns, embed_dim, dropout, max_length
        )
        
        print("\n初始化Transformer:")
        print(f"  blocks={num_blocks}, embed_dim={embed_dim}, num_heads={num_heads}")
        self.transformer = TransformerBlocks(
            num_blocks, embed_dim, num_heads, dropout, lookahead=True
        )
        
        print("")
        self.decoder = Decoder(input_columns, embed_dim)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n总参数数: {total_params:,}")
        print("="*60 + "\n")
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, mask = self.encoder(inputs)
        x = self.transformer(x, mask)
        outputs = self.decoder(x)
        return outputs
    
    def load_converted_weights(self, checkpoint_path: str):
        """加载转换后的权重"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"警告: 缺失 {len(missing_keys)} 个键")
        if unexpected_keys:
            print(f"警告: 多余 {len(unexpected_keys)} 个键")
        
        print("✓ 权重加载完成")


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("测试MFP模型（修复版本）\n")
    
    input_columns = {
        'type': {'is_sequence': True, 'type': 'categorical', 'input_dim': 7, 'shape': [1]},
        'left': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
        'top': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
        'width': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
        'height': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
        'image_embedding': {'is_sequence': True, 'type': 'numerical', 'shape': [512]},
    }
    
    model = MFP(input_columns, embed_dim=256, num_blocks=4)
    
    # 测试前向传播
    batch_size = 2
    seq_len = 10
    
    test_input = {
        'length': torch.tensor([[5], [7]], dtype=torch.long),
        'type': torch.randint(0, 7, (batch_size, seq_len, 1)),
        'left': torch.randint(0, 64, (batch_size, seq_len, 1)),
        'top': torch.randint(0, 64, (batch_size, seq_len, 1)),
        'width': torch.randint(0, 64, (batch_size, seq_len, 1)),
        'height': torch.randint(0, 64, (batch_size, seq_len, 1)),
        'image_embedding': torch.randn(batch_size, seq_len, 512),
    }
    
    print("测试前向传播...")
    with torch.no_grad():
        outputs = model(test_input)
    
    print("\n✓ 前向传播成功!")
    print("\n输出形状:")
    for key, value in outputs.items():
        print(f"  {key:20s}: {list(value.shape)}")
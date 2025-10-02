"""
PyTorch模型架构
MFP (Masked Field Prediction) 模型的PyTorch实现
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
        """
        Args:
            x: (batch, seq_len, embed_dim)
            mask: (batch, seq_len) 布尔掩码，True表示有效位置
        """
        B, S, D = x.shape
        
        # 投影到Q, K, V
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (B, num_heads, S, head_dim)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: (B, num_heads, S, S)
        
        # 应用掩码
        if mask is not None:
            # padding mask
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            scores = scores.masked_fill(~mask, float('-inf'))
            
            # causal mask (如果不允许lookahead)
            if not self.lookahead:
                causal_mask = torch.triu(
                    torch.ones(S, S, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
                scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax和dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        out = torch.matmul(attn_weights, v)  # (B, num_heads, S, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        
        # 输出投影
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
        # Pre-norm架构 (DeepSVG)
        # Self-attention
        residual = x
        x = self.norm1(x)
        x = self.attn(x, mask)
        x = self.dropout1(x)
        x = residual + x
        
        # Feed-forward
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


# ==================== Encoder ====================

class Encoder(nn.Module):
    """编码器：将离散和连续特征编码为序列表示"""
    
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
        
        # 为每个特征创建编码层
        self.embeddings = nn.ModuleDict()
        
        for key, column in input_columns.items():
            if not column.get('is_sequence', False):
                continue
            
            # 检查键是否已存在（避免重复）
            if key in self.embeddings:
                continue
                
            if column['type'] == 'categorical':
                # 分类特征使用Embedding
                vocab_size = column['input_dim'] + 2  # +2 for <MASK> and <NULL>
                self.embeddings[key] = nn.Embedding(vocab_size, embed_dim)
            elif column['type'] == 'numerical':
                # 数值特征使用Linear
                input_size = column['shape'][-1] if 'shape' in column else 1
                self.embeddings[key] = nn.Linear(input_size, embed_dim)
                # 特殊token的embedding
                special_key = f'{key}_special'
                if special_key not in self.embeddings:
                    self.embeddings[special_key] = nn.Embedding(2, embed_dim)
        
        # 位置编码
        self.pos_embedding = nn.Embedding(max_length + 1, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> tuple:
        """
        Args:
            inputs: 特征字典
        
        Returns:
            embeddings: (batch, seq_len, embed_dim)
            mask: (batch, seq_len) 有效位置的掩码
        """
        batch_size = inputs['length'].size(0)
        seq_len = inputs['left'].size(1)  # 使用left作为参考
        
        # 初始化序列表示
        seq_embs = []
        
        # 编码每个特征
        for key, column in self.input_columns.items():
            if not column.get('is_sequence', False):
                continue
            
            if key not in inputs:
                continue
            
            x = inputs[key]
            
            if column['type'] == 'categorical':
                # Embedding查找
                emb = self.embeddings[key](x)  # (B, S, F, D)
                # 对多个特征维度求和 (如RGB的3个通道)
                if len(emb.shape) == 4:
                    emb = emb.sum(dim=2)  # (B, S, D)
            else:
                # 数值特征的线性变换
                emb = self.embeddings[key](x)  # (B, S, D)
            
            seq_embs.append(emb)
        
        # 融合所有特征 (加法融合)
        seq = torch.stack(seq_embs).sum(dim=0)  # (B, S, D)
        
        # 添加位置编码
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        seq = seq + self.pos_embedding(positions)
        seq = self.dropout(seq)
        
        # 生成掩码
        lengths = inputs['length'].squeeze(-1)  # (B,)
        mask = torch.arange(seq_len, device=seq.device).unsqueeze(0) < lengths.unsqueeze(1)
        
        return seq, mask


# ==================== Decoder ====================

class Decoder(nn.Module):
    """解码器：将序列表示解码为各个特征的预测"""
    
    def __init__(self, input_columns: Dict, embed_dim: int = 128):
        super().__init__()
        self.input_columns = input_columns
        
        # 为每个特征创建解码头
        self.heads = nn.ModuleDict()
        
        for key, column in input_columns.items():
            if not column.get('is_sequence', False):
                continue
            
            if column['type'] == 'categorical':
                # 分类特征：输出logits
                shape = column.get('shape', [1])
                output_dim = shape[-1] * column['input_dim']
                self.heads[key] = nn.Linear(embed_dim, output_dim)
            else:
                # 数值特征：直接回归
                shape = column.get('shape', [1])
                output_dim = shape[-1]
                self.heads[key] = nn.Linear(embed_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, embed_dim)
        
        Returns:
            outputs: 各特征的预测字典
        """
        outputs = {}
        batch_size, seq_len, _ = x.shape
        
        for key, column in self.input_columns.items():
            if not column.get('is_sequence', False):
                continue
            
            pred = self.heads[key](x)  # (B, S, output_dim)
            
            if column['type'] == 'categorical':
                # Reshape为 (B, S, num_features, vocab_size)
                shape = column.get('shape', [1])
                num_features = shape[-1]
                vocab_size = column['input_dim']
                pred = pred.view(batch_size, seq_len, num_features, vocab_size)
            
            outputs[key] = pred
        
        return outputs


# ==================== MFP Model ====================

class MFP(nn.Module):
    """
    Masked Field Prediction模型
    基于Transformer的布局生成模型
    """
    
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
        
        # 编码器
        self.encoder = Encoder(
            input_columns, embed_dim, dropout, max_length
        )
        
        # Transformer块
        self.transformer = TransformerBlocks(
            num_blocks, embed_dim, num_heads, dropout, lookahead=True
        )
        
        # 解码器
        self.decoder = Decoder(input_columns, embed_dim)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs: 输入特征字典
        
        Returns:
            outputs: 预测输出字典
        """
        # 编码
        x, mask = self.encoder(inputs)
        
        # Transformer处理
        x = self.transformer(x, mask)
        
        # 解码
        outputs = self.decoder(x)
        
        return outputs
    
    def load_converted_weights(self, checkpoint_path: str):
        """加载转换后的权重"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # 尝试加载权重
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"警告: 缺失的键: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"警告: 多余的键: {unexpected_keys[:5]}...")
        
        print("✓ 权重加载完成")


# 测试代码
if __name__ == "__main__":
    # 构造示例输入列
    input_columns = {
        'type': {'is_sequence': True, 'type': 'categorical', 'input_dim': 10, 'shape': [1]},
        'left': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
        'top': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
        'width': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
        'height': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
        'color': {'is_sequence': True, 'type': 'categorical', 'input_dim': 16, 'shape': [3]},
        'image_embedding': {'is_sequence': True, 'type': 'numerical', 'shape': [512]},
    }
    
    # 创建模型
    model = MFP(input_columns, embed_dim=128, num_blocks=4)
    print(f"模型参数数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建示例输入
    batch_size = 4
    seq_len = 10
    
    inputs = {
        'length': torch.tensor([[8], [10], [5], [7]], dtype=torch.long),
        'type': torch.randint(0, 10, (batch_size, seq_len, 1)),
        'left': torch.randint(0, 64, (batch_size, seq_len, 1)),
        'top': torch.randint(0, 64, (batch_size, seq_len, 1)),
        'width': torch.randint(0, 64, (batch_size, seq_len, 1)),
        'height': torch.randint(0, 64, (batch_size, seq_len, 1)),
        'color': torch.randint(0, 16, (batch_size, seq_len, 3)),
        'image_embedding': torch.randn(batch_size, seq_len, 512),
    }
    
    # 前向传播
    outputs = model(inputs)
    
    print("\n输出形状:")
    for key, value in outputs.items():
        print(f"  {key:20s}: {list(value.shape)}")
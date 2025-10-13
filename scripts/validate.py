"""
模型验证脚本
用于评估训练好的模型性能
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import numpy as np
from typing import Dict

from dataset import create_dataloader
from models_pytorch import MFP


class ModelValidator:
    """模型验证器"""
    
    def __init__(self, model, val_loader, device='cuda'):
        self.model = model.to(device)
        self.val_loader = val_loader
        self.device = device
        self.model.eval()
    
    def compute_metrics(self, predictions, targets, mask):
        """计算评估指标"""
        metrics = {}
        
        for key in predictions.keys():
            if key not in targets:
                continue
            
            pred = predictions[key]
            target = targets[key]
            column = self.model.input_columns.get(key, {})
            
            if column.get('type') == 'categorical':
                # 分类准确率
                if pred.dim() == 4:  # (B, S, F, C)
                    B, S, F, C = pred.shape
                    pred_labels = pred.argmax(dim=-1)  # (B, S, F)
                    target = target.long()
                    
                    mask_expanded = mask.unsqueeze(-1).expand(B, S, F)
                    correct = (pred_labels == target) & mask_expanded
                    accuracy = correct.float().sum() / mask_expanded.sum()
                    
                elif pred.dim() == 3:  # (B, S, C)
                    pred_labels = pred.argmax(dim=-1)  # (B, S)
                    target = target.squeeze(-1).long()
                    
                    correct = (pred_labels == target) & mask
                    accuracy = correct.float().sum() / mask.sum()
                
                metrics[f'{key}_accuracy'] = accuracy.item()
                
            elif column.get('type') == 'numerical':
                # MSE和MAE
                mask_expanded = mask.unsqueeze(-1).expand_as(pred)
                mse = ((pred - target) ** 2 * mask_expanded.float()).sum() / mask_expanded.sum()
                mae = ((pred - target).abs() * mask_expanded.float()).sum() / mask_expanded.sum()
                
                metrics[f'{key}_mse'] = mse.item()
                metrics[f'{key}_mae'] = mae.item()
        
        return metrics
    
    @torch.no_grad()
    def validate(self):
        """执行完整验证"""
        all_metrics = {}
        
        print("开始验证...")
        for batch in tqdm(self.val_loader):
            # 移动到设备
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v 
                     for k, v in batch.items()}
            
            # 前向传播
            outputs = self.model(inputs)
            
            # 生成掩码
            lengths = inputs['length'].squeeze(-1)
            max_len = inputs['left'].size(1)
            mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
            
            # 计算指标
            metrics = self.compute_metrics(outputs, inputs, mask)
            
            # 累积指标
            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
        
        # 计算平均指标
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        
        return avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Validate MFP Model')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据目录')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--config', type=str, default='config/train_config.json',
                       help='训练配置文件')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='验证数据集划分')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # 加载配置
    print("加载配置...")
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 创建数据加载器
    print(f"加载 {args.split} 数据集...")
    val_loader = create_dataloader(
        args.data_dir, args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        max_length=config['data'].get('max_length', 20),
        bins=config['data'].get('bins', 64),
    )
    
    # 获取input_columns
    input_columns = val_loader.dataset.get_input_columns()
    
    # 加载模型
    print("加载模型...")
    model = MFP(
        input_columns=input_columns,
        embed_dim=config['model']['embed_dim'],
        num_blocks=config['model']['num_blocks'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        max_length=config['model']['max_length'],
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
    model.load_state_dict(state_dict, strict=False)
    
    print(f"✓ 模型加载完成")
    print(f"  Checkpoint: {args.checkpoint}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    
    # 创建验证器
    validator = ModelValidator(model, val_loader, args.device)
    
    # 执行验证
    metrics = validator.validate()
    
    # 打印结果
    print("\n" + "="*60)
    print("验证结果")
    print("="*60)
    for key, value in sorted(metrics.items()):
        print(f"{key:30s}: {value:.6f}")
    print("="*60)
    
    # 保存结果
    results_path = Path(args.checkpoint).parent / f'validation_results_{args.split}.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ 结果保存到: {results_path}")


if __name__ == "__main__":
    main()
    
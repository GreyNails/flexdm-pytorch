"""
改进的PyTorch训练脚本
修复了原始代码的问题并添加更多功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import time
import os
from typing import Dict

from dataset import create_dataloader, DesignLayoutDataset
from models_pytorch import MFP


class ImprovedMFPTrainer:
    """改进的MFP模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        save_dir: str = './checkpoints',
        log_dir: str = './logs',
        resume_path: str = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # 训练配置
        train_cfg = config.get('training', {})
        self.num_epochs = train_cfg.get('num_epochs', 100)
        self.gradient_clip = train_cfg.get('gradient_clip', 1.0)
        self.accumulation_steps = train_cfg.get('accumulation_steps', 1)
        
        # 损失权重
        self.loss_weights = config.get('loss_weights', {})
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=train_cfg.get('learning_rate', 1e-4),
            betas=(0.9, 0.999),
            weight_decay=train_cfg.get('weight_decay', 0.01),
        )
        
        # 学习率调度器
        scheduler_cfg = config.get('scheduler', {})
        if scheduler_cfg.get('type') == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_cfg.get('mode', 'min'),
                factor=scheduler_cfg.get('factor', 0.5),
                patience=scheduler_cfg.get('patience', 5),
                min_lr=scheduler_cfg.get('min_lr', 1e-6),
                verbose=True,
            )
        else:
            self.scheduler = None
        
        # 目录设置
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir)
        
        # 训练状态
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # 恢复训练
        if resume_path and Path(resume_path).exists():
            self.load_checkpoint(resume_path)
        
        print(f"✓ 训练器初始化完成")
        print(f"  设备: {device}")
        print(f"  训练批次: {len(train_loader)}")
        print(f"  验证批次: {len(val_loader)}")
        print(f"  开始epoch: {self.start_epoch}")
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失
        
        Args:
            predictions: 模型预测
            targets: 真实标签
            mask: 有效位置掩码 (B, S)
        
        Returns:
            损失字典
        """
        losses = {}
        total_loss = 0.0
        
        for key in predictions.keys():
            if key not in targets:
                continue
            
            pred = predictions[key]
            target = targets[key]
            
            # 获取列信息
            column = self.model.input_columns.get(key, {})
            weight = self.loss_weights.get(key, 1.0)
            
            if column.get('type') == 'categorical':
                # 分类任务：交叉熵损失
                # pred: (B, S, num_feat, C), target: (B, S, num_feat)
                if pred.dim() == 4:  # (B, S, num_feat, C)
                    B, S, num_feat, C = pred.shape
                    pred = pred.reshape(B * S * num_feat, C)
                    target = target.reshape(B * S * num_feat).long()
                    
                    loss = F.cross_entropy(pred, target, reduction='none')
                    loss = loss.reshape(B, S, num_feat)
                    
                    # 应用掩码并求平均
                    mask_expanded = mask.unsqueeze(-1).expand_as(loss)
                    loss = (loss * mask_expanded.float()).sum() / (mask_expanded.sum() + 1e-8)
                    
                elif pred.dim() == 3:  # (B, S, C)
                    B, S, C = pred.shape
                    pred = pred.reshape(B * S, C)
                    target = target.reshape(B * S).long()
                    
                    loss = F.cross_entropy(pred, target, reduction='none')
                    loss = loss.reshape(B, S)
                    
                    # 应用掩码
                    loss = (loss * mask.float()).sum() / (mask.sum() + 1e-8)
                else:
                    continue
            
            elif column.get('type') == 'numerical':
                # 回归任务：MSE损失
                # pred: (B, S, D), target: (B, S, D)
                loss = F.mse_loss(pred, target, reduction='none')
                
                # 应用掩码
                mask_expanded = mask.unsqueeze(-1).expand_as(loss)
                loss = (loss * mask_expanded.float()).sum() / (mask_expanded.sum() + 1e-8)
            else:
                continue
            
            # 应用权重
            weighted_loss = loss * weight
            losses[f'{key}_loss'] = loss.detach()
            total_loss += weighted_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        epoch_losses = {}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.num_epochs}')
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # 移动到设备
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v 
                     for k, v in batch.items()}
            
            # 前向传播
            outputs = self.model(inputs)
            
            # 生成掩码
            lengths = inputs['length'].squeeze(-1)
            max_len = inputs['left'].size(1)
            mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
            
            # 计算损失
            losses = self.compute_loss(outputs, inputs, mask)
            loss = losses['total_loss'] / self.accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # 记录损失
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value.item() if torch.is_tensor(value) else value)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # TensorBoard记录
            log_cfg = self.config.get('logging', {})
            if self.global_step % log_cfg.get('log_interval', 100) == 0:
                for key, value in losses.items():
                    self.writer.add_scalar(
                        f'train/{key}', 
                        value.item() if torch.is_tensor(value) else value, 
                        self.global_step
                    )
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        # 计算epoch平均损失
        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """验证"""
        self.model.eval()
        epoch_losses = {}
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            # 移动到设备
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v 
                     for k, v in batch.items()}
            
            # 前向传播
            outputs = self.model(inputs)
            
            # 生成掩码
            lengths = inputs['length'].squeeze(-1)
            max_len = inputs['left'].size(1)
            mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
            
            # 计算损失
            losses = self.compute_loss(outputs, inputs, mask)
            
            # 记录损失
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                val = value.item() if torch.is_tensor(value) else value
                epoch_losses[key].append(val)
        
        # 计算平均损失
        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        
        # TensorBoard记录
        for key, value in avg_losses.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)
        
        return avg_losses
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'val_loss': val_loss,
            'config': self.config,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新模型
        torch.save(checkpoint, self.save_dir / 'latest.pth')
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pth')
            print(f"✓ 保存最佳模型 (epoch {epoch}, val_loss: {val_loss:.4f})")
        
        # 定期保存
        ckpt_cfg = self.config.get('checkpoint', {})
        if epoch % ckpt_cfg.get('save_interval', 5) == 0:
            torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pth')
        
        # 清理旧checkpoint
        self._cleanup_checkpoints(ckpt_cfg.get('keep_last_n', 3))
    
    def _cleanup_checkpoints(self, keep_last_n: int):
        """清理旧的checkpoint"""
        checkpoints = sorted(self.save_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) > keep_last_n:
            for ckpt in checkpoints[:-keep_last_n]:
                ckpt.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载checkpoint"""
        print(f"加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"✓ 恢复训练从 epoch {self.start_epoch}")
    
    def train(self):
        """完整训练流程"""
        print("\n" + "="*60)
        print("开始训练")
        print("="*60)
        
        for epoch in range(self.start_epoch, self.num_epochs):
            start_time = time.time()
            
            # 训练
            train_losses = self.train_epoch(epoch)
            
            # 验证
            log_cfg = self.config.get('logging', {})
            if epoch % log_cfg.get('val_interval', 1) == 0:
                val_losses = self.validate(epoch)
            else:
                val_losses = {'total_loss': float('inf')}
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step(val_losses['total_loss'])
            
            # 打印信息
            epoch_time = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.num_epochs - 1} ({epoch_time:.1f}s)")
            print(f"{'='*60}")
            print(f"Train Loss: {train_losses['total_loss']:.4f}")
            if 'total_loss' in val_losses and val_losses['total_loss'] != float('inf'):
                print(f"Val Loss:   {val_losses['total_loss']:.4f}")
            print(f"LR:         {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"Best Val:   {self.best_val_loss:.4f}")
            
            # 保存checkpoint
            is_best = val_losses['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total_loss']
            
            self.save_checkpoint(epoch, val_losses['total_loss'], is_best)
        
        print("\n" + "="*60)
        print("训练完成!")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print("="*60)
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train MFP Model')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                       help='JSON数据目录')
    parser.add_argument('--config', type=str, default='config/train_config.json',
                       help='训练配置文件')
    
    # 覆盖配置的参数
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--embed_dim', type=int, default=None)
    parser.add_argument('--num_blocks', type=int, default=None)
    parser.add_argument('--num_heads', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    
    # 训练参数
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的checkpoint路径')
    
    args = parser.parse_args()
    
    # 加载配置
    print("加载配置...")
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 命令行参数覆盖配置
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.num_epochs is not None:
        config['training']['num_epochs'] = args.num_epochs
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.embed_dim is not None:
        config['model']['embed_dim'] = args.embed_dim
    if args.num_blocks is not None:
        config['model']['num_blocks'] = args.num_blocks
    if args.num_heads is not None:
        config['model']['num_heads'] = args.num_heads
    if args.num_workers is not None:
        config['data']['num_workers'] = args.num_workers
    
    print(json.dumps(config, indent=2))
    
    # 创建数据加载器
    print("\n加载数据...")
    train_loader = create_dataloader(
        args.data_dir, 'train',
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        max_length=config['data'].get('max_length', 20),
        bins=config['data'].get('bins', 64),
        min_font_freq=config['data'].get('min_font_freq', 500),
    )
    
    val_loader = create_dataloader(
        args.data_dir, 'val',
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        max_length=config['data'].get('max_length', 20),
        bins=config['data'].get('bins', 64),
        min_font_freq=config['data'].get('min_font_freq', 500),
    )
    
    # 获取input_columns
    dataset = train_loader.dataset
    input_columns = dataset.get_input_columns()
    
    # 保存input_columns
    input_columns_path = Path(args.save_dir).parent / 'input_columns_used.json'
    input_columns_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_columns_path, 'w') as f:
        json.dump(input_columns, f, indent=2)
    print(f"✓ Input columns保存到: {input_columns_path}")
    
    # 创建模型
    print("\n创建模型...")
    model = MFP(
        input_columns=input_columns,
        embed_dim=config['model']['embed_dim'],
        num_blocks=config['model']['num_blocks'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        max_length=config['model']['max_length'],
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 创建训练器
    trainer = ImprovedMFPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        resume_path=args.resume,
    )
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
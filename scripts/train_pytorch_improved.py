"""
完整的MFP训练代码 - 严格对齐TensorFlow版本
包含正确的Masking机制
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
from typing import Dict
import numpy as np

from dataset import create_dataloader
from models_pytorch import MFP
from masking_pytorch import (
    get_task_names,
    preprocess_for_train,
    get_seq_mask,
)


class MFPTrainer:
    """MFP模型训练器（包含Masking）"""
    
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
        self.input_columns = model.input_columns
        
        # 训练配置
        train_cfg = config.get('training', {})
        self.num_epochs = train_cfg.get('num_epochs', 100)
        self.gradient_clip = train_cfg.get('gradient_clip', 1.0)
        self.accumulation_steps = train_cfg.get('accumulation_steps', 1)
        
        # 任务采样
        self.task_names = get_task_names(self.input_columns)
        self.num_tasks = len(self.task_names)
        print(f"\n任务列表: {self.task_names}")
        
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
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True,
        )
        
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
        
        print(f"✓ 训练器初始化完成 (设备: {device})")
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失（只在masked位置计算）
        
        Args:
            predictions: 模型预测
            targets: 真实标签
            masks: mask字典（True表示需要预测的位置）
            seq_mask: (B, S) 有效位置mask
        
        Returns:
            损失字典
        """
        losses = {}
        total_loss = 0.0
        
        for key in predictions.keys():
            if key not in targets or key not in masks:
                continue
            
            pred = predictions[key]
            target = targets[key]
            mfp_mask = masks[key]
            
            column = self.input_columns.get(key, {})
            weight = self.loss_weights.get(key, 1.0)
            
            if column.get('type') == 'categorical':
                # ⭐ 关键修复：过滤掉特殊token
                input_dim = column['input_dim']
                
                if pred.dim() == 4:  # (B, S, num_feat, C)
                    B, S, num_feat, C = pred.shape
                    
                    # 展平
                    pred_flat = pred.reshape(B * S * num_feat, C)
                    target_flat = target.reshape(B * S * num_feat).long()
                    
                    # ⭐ 将超出范围的目标值clip到有效范围
                    # target_flat = torch.clamp(target_flat, 0, input_dim - 1)
                    
                    # 计算损失
                    loss = F.cross_entropy(pred_flat, target_flat, reduction='none')
                    loss = loss.reshape(B, S, num_feat)
                    
                    # 应用mask
                    mask_expanded = mfp_mask.unsqueeze(-1) & seq_mask.unsqueeze(-1)
                    loss = (loss * mask_expanded.float()).sum() / (mask_expanded.sum() + 1e-8)
                
                elif pred.dim() == 3:  # (B, S, C)
                    B, S, C = pred.shape
                    
                    # 展平
                    pred_flat = pred.reshape(B * S, C)
                    target_flat = target.reshape(B * S).long()
                    
                    # ⭐ 将超出范围的目标值clip到有效范围
                    # target_flat = torch.clamp(target_flat, 0, input_dim - 1)
                    
                    # 计算损失
                    loss = F.cross_entropy(pred_flat, target_flat, reduction='none')
                    loss = loss.reshape(B, S)
                    
                    # 应用mask
                    mask_combined = mfp_mask & seq_mask
                    loss = (loss * mask_combined.float()).sum() / (mask_combined.sum() + 1e-8)
                else:
                    continue
            
            elif column.get('type') == 'numerical':
                # 回归损失
                loss = F.mse_loss(pred, target.float(), reduction='none')
                
                # 应用mask
                mask_expanded = mfp_mask.unsqueeze(-1) & seq_mask.unsqueeze(-1)
                mask_expanded = mask_expanded.expand_as(loss)
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
        """训练一个epoch（包含Masking）"""
        self.model.train()
        epoch_losses = {}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.num_epochs}')
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # 移动到设备
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v 
                     for k, v in batch.items()}
            
            # 随机选择任务
            batch_size = inputs['length'].size(0)
            task_ids = torch.randint(0, self.num_tasks, (batch_size,))
            
            # 预处理（应用Masking）
            targets, modified_inputs, masks = preprocess_for_train(
                inputs,
                self.input_columns,
                task_ids[0].item(),  # 简化：整个batch使用同一任务
                is_autoreg=False,
            )
            
            # 前向传播
            outputs = self.model(modified_inputs)
            
            # 生成序列mask
            seq_mask = get_seq_mask(inputs['length'], max_len=inputs['left'].size(1))
            
            # 计算损失
            losses = self.compute_loss(outputs, targets, masks, seq_mask)
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
            if self.global_step % 100 == 0:
                for key, value in losses.items():
                    self.writer.add_scalar(
                        f'train/{key}', 
                        value.item() if torch.is_tensor(value) else value, 
                        self.global_step
                    )
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        # 计算平均损失
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
            
            # 随机选择任务
            batch_size = inputs['length'].size(0)
            task_ids = torch.randint(0, self.num_tasks, (batch_size,))
            
            # 预处理
            targets, modified_inputs, masks = preprocess_for_train(
                inputs,
                self.input_columns,
                task_ids[0].item(),
                is_autoreg=False,
            )
            
            # 前向传播
            outputs = self.model(modified_inputs)
            
            # 生成序列mask
            seq_mask = get_seq_mask(inputs['length'], max_len=inputs['left'].size(1))
            
            # 计算损失
            losses = self.compute_loss(outputs, targets, masks, seq_mask)
            
            # 记录损失
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value.item() if torch.is_tensor(value) else value)
        
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
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'val_loss': val_loss,
            'config': self.config,
        }
        
        # 保存最新模型
        torch.save(checkpoint, self.save_dir / 'latest.pth')
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pth')
            print(f"✓ 保存最佳模型 (epoch {epoch}, val_loss: {val_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载checkpoint"""
        print(f"加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"✓ 恢复训练从 epoch {self.start_epoch}")
    
    def train(self):
        """完整训练流程"""
        print("\n" + "="*60)
        print("开始训练（包含Masking机制）")
        print("="*60)
        
        for epoch in range(self.start_epoch, self.num_epochs):
            start_time = time.time()
            
            # 训练
            train_losses = self.train_epoch(epoch)
            
            # 验证
            if epoch % 1 == 0:
                val_losses = self.validate(epoch)
            else:
                val_losses = {'total_loss': float('inf')}
            
            # 学习率调度
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
    parser = argparse.ArgumentParser(description='Train MFP Model with Masking')
    
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/train_config.json')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 命令行参数覆盖
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.num_epochs is not None:
        config['training']['num_epochs'] = args.num_epochs
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    
    # 创建数据加载器
    print("\n加载数据...")
    train_loader = create_dataloader(
        args.data_dir, 'train',
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        max_length=config['data'].get('max_length', 20),
    )
    
    val_loader = create_dataloader(
        args.data_dir, 'val',
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        max_length=config['data'].get('max_length', 20),
    )
    
    # 获取input_columns
    dataset = train_loader.dataset
    input_columns = dataset.get_input_columns()
    
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
    
    # 创建训练器
    trainer = MFPTrainer(
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
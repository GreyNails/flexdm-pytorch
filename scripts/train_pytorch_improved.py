"""
完整的MFP训练代码 - 严格对齐TensorFlow版本
包含正确的Masking机制和Loss曲线绘制
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
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import seaborn as sns
sns.set_style('darkgrid')

from dataset import create_dataloader
from models_pytorch import MFP
from masking_pytorch import (
    get_task_names,
    preprocess_for_train,
    get_seq_mask,
)


class MFPTrainer:
    """MFP模型训练器（包含Masking和Loss曲线绘制）"""
    
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
        self.plot_dir = self.save_dir / 'plots'
        self.plot_dir.mkdir(exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir)
        
        # 训练状态
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # 🎨 Loss历史记录 - 用于绘图
        self.loss_history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_losses_detailed': {},  # 每个任务的训练损失
            'val_losses_detailed': {},    # 每个任务的验证损失
            'learning_rates': [],
        }
        
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
    
    def plot_loss_curves(self, save_path: str = None):
        """
        绘制损失曲线
        
        Args:
            save_path: 保存路径，如果为None则保存到默认位置
        """
        if len(self.loss_history['epochs']) == 0:
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        # 1. 主要损失曲线（训练vs验证）
        ax1 = axes[0, 0]
        ax1.plot(self.loss_history['epochs'], self.loss_history['train_loss'], 
                'b-', label='Train Loss', linewidth=2)
        ax1.plot(self.loss_history['epochs'], self.loss_history['val_loss'], 
                'r-', label='Val Loss', linewidth=2)
        ax1.fill_between(self.loss_history['epochs'], 
                        self.loss_history['train_loss'], 
                        self.loss_history['val_loss'], 
                        alpha=0.2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training vs Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 标记最佳验证损失
        if self.loss_history['val_loss']:
            best_epoch = np.argmin(self.loss_history['val_loss'])
            best_val_loss = self.loss_history['val_loss'][best_epoch]
            ax1.plot(self.loss_history['epochs'][best_epoch], best_val_loss, 
                    'g*', markersize=15, label=f'Best Val Loss: {best_val_loss:.4f}')
            ax1.legend()
        
        # 2. 学习率曲线
        ax2 = axes[0, 1]
        ax2.plot(self.loss_history['epochs'], self.loss_history['learning_rates'], 
                'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 3. 每个任务的训练损失
        ax3 = axes[1, 0]
        for key in self.loss_history['train_losses_detailed'].keys():
            if key != 'total_loss' and len(self.loss_history['train_losses_detailed'][key]) > 0:
                ax3.plot(self.loss_history['epochs'], 
                        self.loss_history['train_losses_detailed'][key], 
                        label=key.replace('_loss', ''), linewidth=1.5, alpha=0.8)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss by Task')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. 每个任务的验证损失
        ax4 = axes[1, 1]
        for key in self.loss_history['val_losses_detailed'].keys():
            if key != 'total_loss' and len(self.loss_history['val_losses_detailed'][key]) > 0:
                ax4.plot(self.loss_history['epochs'], 
                        self.loss_history['val_losses_detailed'][key], 
                        label=key.replace('_loss', ''), linewidth=1.5, alpha=0.8)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.set_title('Validation Loss by Task')
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_path = self.plot_dir / f'loss_curves_epoch_{len(self.loss_history["epochs"])}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Loss曲线已保存至: {save_path}")
    
    def plot_loss_comparison(self):
        """绘制训练和验证损失的对比图（简化版）"""
        if len(self.loss_history['epochs']) == 0:
            return
        
        plt.figure(figsize=(10, 6))
        
        # 绘制损失曲线
        plt.plot(self.loss_history['epochs'], self.loss_history['train_loss'], 
                'b-', label='Train Loss', linewidth=2, alpha=0.8)
        plt.plot(self.loss_history['epochs'], self.loss_history['val_loss'], 
                'r-', label='Val Loss', linewidth=2, alpha=0.8)
        
        # 添加最佳点标记
        if self.loss_history['val_loss']:
            best_epoch = np.argmin(self.loss_history['val_loss'])
            best_val_loss = self.loss_history['val_loss'][best_epoch]
            plt.scatter(self.loss_history['epochs'][best_epoch], best_val_loss, 
                       color='green', s=100, zorder=5, 
                       label=f'Best: {best_val_loss:.4f}')
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Progress', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # 保存简化版图片
        save_path = self.plot_dir / 'loss_curve_simple.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def update_loss_history(self, epoch: int, train_losses: Dict, val_losses: Dict):
        """更新损失历史记录"""
        self.loss_history['epochs'].append(epoch)
        self.loss_history['train_loss'].append(train_losses['total_loss'])
        self.loss_history['val_loss'].append(val_losses['total_loss'])
        self.loss_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
        
        # 更新详细损失
        for key, value in train_losses.items():
            if key not in self.loss_history['train_losses_detailed']:
                self.loss_history['train_losses_detailed'][key] = []
            self.loss_history['train_losses_detailed'][key].append(value)
        
        for key, value in val_losses.items():
            if key not in self.loss_history['val_losses_detailed']:
                self.loss_history['val_losses_detailed'][key] = []
            self.loss_history['val_losses_detailed'][key].append(value)
    
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
        """保存checkpoint（包含loss历史）"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'val_loss': val_loss,
            'config': self.config,
            'loss_history': self.loss_history,  # 🎨 保存损失历史
        }
        
        # 保存最新模型
        torch.save(checkpoint, self.save_dir / 'latest.pth')
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pth')
            print(f"✓ 保存最佳模型 (epoch {epoch}, val_loss: {val_loss:.4f})")
        
        # 🎨 保存损失历史到JSON（方便外部分析）
        with open(self.save_dir / 'loss_history.json', 'w') as f:
            json.dump(self.loss_history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载checkpoint（包含loss历史）"""
        print(f"加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # 🎨 恢复损失历史
        if 'loss_history' in checkpoint:
            self.loss_history = checkpoint['loss_history']
            print(f"✓ 恢复损失历史 ({len(self.loss_history['epochs'])} epochs)")
        
        print(f"✓ 恢复训练从 epoch {self.start_epoch}")
    
    def train(self):
        """完整训练流程（包含loss曲线绘制）"""
        print("\n" + "="*60)
        print("开始训练（包含Masking机制和Loss曲线绘制）")
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
            
            # 🎨 更新损失历史
            self.update_loss_history(epoch, train_losses, val_losses)
            
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
            
            # 🎨 定期绘制损失曲线
            if epoch % 5 == 0 or epoch == self.num_epochs - 1 or is_best:
                self.plot_loss_curves()
                self.plot_loss_comparison()
        
        # 🎨 训练结束，绘制最终损失曲线
        print("\n📊 绘制最终损失曲线...")
        self.plot_loss_curves(self.plot_dir / 'final_loss_curves.png')
        self.plot_loss_comparison()
        
        print("\n" + "="*60)
        print("训练完成!")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print(f"损失曲线已保存至: {self.plot_dir}")
        print("="*60)
        self.writer.close()


def visualize_training_summary(checkpoint_path: str):
    """从checkpoint可视化训练总结"""
    print("\n📊 生成训练总结...")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    loss_history = checkpoint.get('loss_history', {})
    
    if not loss_history or not loss_history.get('epochs'):
        print("❌ 没有找到损失历史数据")
        return
    
    # 创建总结图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Summary', fontsize=16, fontweight='bold')
    
    # 1. 损失曲线
    ax = axes[0, 0]
    ax.plot(loss_history['epochs'], loss_history['train_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(loss_history['epochs'], loss_history['val_loss'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 损失差距
    ax = axes[0, 1]
    gap = np.array(loss_history['val_loss']) - np.array(loss_history['train_loss'])
    ax.plot(loss_history['epochs'], gap, 'g-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gap (Val - Train)')
    ax.set_title('Generalization Gap')
    ax.grid(True, alpha=0.3)
    
    # 3. 学习率
    ax = axes[0, 2]
    ax.plot(loss_history['epochs'], loss_history['learning_rates'], 'orange', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 4. 损失分布（箱线图）
    ax = axes[1, 0]
    data_to_plot = [loss_history['train_loss'][-20:], loss_history['val_loss'][-20:]]
    bp = ax.boxplot(data_to_plot, labels=['Train', 'Val'])
    ax.set_ylabel('Loss')
    ax.set_title('Loss Distribution (Last 20 Epochs)')
    ax.grid(True, alpha=0.3)
    
    # 5. 收敛速度
    ax = axes[1, 1]
    if len(loss_history['val_loss']) > 1:
        convergence_rate = np.diff(loss_history['val_loss'])
        ax.plot(loss_history['epochs'][1:], convergence_rate, 'purple', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Change')
        ax.set_title('Convergence Rate (Val Loss)')
        ax.grid(True, alpha=0.3)
    
    # 6. 最佳模型信息
    ax = axes[1, 2]
    ax.axis('off')
    best_epoch = np.argmin(loss_history['val_loss'])
    best_val_loss = loss_history['val_loss'][best_epoch]
    best_train_loss = loss_history['train_loss'][best_epoch]
    
    info_text = f"""
    Best Model Information:
    ----------------------
    Epoch: {loss_history['epochs'][best_epoch]}
    Val Loss: {best_val_loss:.4f}
    Train Loss: {best_train_loss:.4f}
    Gap: {best_val_loss - best_train_loss:.4f}
    
    Final Model:
    -----------
    Epoch: {loss_history['epochs'][-1]}
    Val Loss: {loss_history['val_loss'][-1]:.4f}
    Train Loss: {loss_history['train_loss'][-1]:.4f}
    """
    ax.text(0.1, 0.5, info_text, fontsize=11, family='monospace', 
            verticalalignment='center')
    
    plt.tight_layout()
    
    # 保存
    save_path = Path(checkpoint_path).parent / 'training_summary.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 训练总结已保存至: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train MFP Model with Masking and Loss Visualization')
    
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/train_config.json')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize_only', type=str, default=None, 
                       help='仅可视化已有的checkpoint')
    
    args = parser.parse_args()
    
    # 如果只是可视化
    if args.visualize_only:
        visualize_training_summary(args.visualize_only)
        return
    
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
    
    # 生成最终的训练总结
    final_checkpoint = Path(args.save_dir) / 'best.pth'
    if final_checkpoint.exists():
        visualize_training_summary(str(final_checkpoint))


if __name__ == "__main__":
    main()
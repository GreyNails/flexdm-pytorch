"""
PyTorch训练脚本
MFP模型的训练和评估
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import time

from dataset import create_dataloader
from models_pytorch import MFP


class MFPTrainer:
    """MFP模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        save_dir: str = './checkpoints',
        log_dir: str = './logs',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
        )
        
        # 目录设置
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir)
        
        # 跟踪最佳模型
        self.best_val_loss = float('inf')
        self.global_step = 0
    
    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        mask: torch.Tensor,
    ) -> dict:
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
            pred = predictions[key]
            target = targets[key]
            
            # 获取列信息
            column = self.model.input_columns[key]
            
            if column['type'] == 'categorical':
                # 分类任务：交叉熵损失
                # pred: (B, S, F, C), target: (B, S, F)
                B, S, F, C = pred.shape
                pred = pred.reshape(B * S * F, C)
                target = target.reshape(B * S * F)
                
                loss = F.cross_entropy(pred, target, reduction='none')
                loss = loss.reshape(B, S, F)
                
                # 应用掩码并求平均
                mask_expanded = mask.unsqueeze(-1).expand_as(loss)
                loss = (loss * mask_expanded.float()).sum() / mask_expanded.sum()
            else:
                # 回归任务：MSE损失
                # pred: (B, S, D), target: (B, S, D)
                loss = F.mse_loss(pred, target, reduction='none')
                
                # 应用掩码
                mask_expanded = mask.unsqueeze(-1).expand_as(loss)
                loss = (loss * mask_expanded.float()).sum() / mask_expanded.sum()
            
            losses[f'{key}_loss'] = loss
            total_loss += loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        epoch_losses = {}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
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
            
            # 反向传播
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 记录损失
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value.item())
            
            # 更新进度条
            pbar.set_postfix({'loss': f"{losses['total_loss'].item():.4f}"})
            
            # TensorBoard记录
            if self.global_step % 100 == 0:
                for key, value in losses.items():
                    self.writer.add_scalar(f'train/{key}', value.item(), self.global_step)
            
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
                epoch_losses[key].append(value.item())
        
        # 计算平均损失
        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        
        # TensorBoard记录
        for key, value in avg_losses.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)
        
        return avg_losses
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        # 保存最新模型
        torch.save(checkpoint, self.save_dir / 'latest.pth')
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pth')
            print(f"✓ 保存最佳模型 (epoch {epoch})")
    
    def train(self):
        """完整训练流程"""
        print("开始训练...")
        print(f"设备: {self.device}")
        print(f"训练批次: {len(self.train_loader)}")
        print(f"验证批次: {len(self.val_loader)}")
        
        for epoch in range(1, self.num_epochs + 1):
            start_time = time.time()
            
            # 训练
            train_losses = self.train_epoch(epoch)
            
            # 验证
            val_losses = self.validate(epoch)
            
            # 学习率调度
            self.scheduler.step(val_losses['total_loss'])
            
            # 打印信息
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch}/{self.num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_losses['total_loss']:.4f}")
            print(f"  Val Loss:   {val_losses['total_loss']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存checkpoint
            is_best = val_losses['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total_loss']
            
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        print("\n训练完成!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                       help='JSON数据目录')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    
    # 预训练权重
    parser.add_argument('--pretrained', type=str, default=None,
                       help='预训练权重路径')
    
    args = parser.parse_args()
    
    # 创建数据加载器
    print("加载数据...")
    train_loader = create_dataloader(
        args.data_dir, 'train', args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )
    val_loader = create_dataloader(
        args.data_dir, 'val', args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )
    
    # 获取input_columns (从数据集推断)
    sample = next(iter(train_loader))
    input_columns = {
        'type': {'is_sequence': True, 'type': 'categorical', 'input_dim': 10, 'shape': [1]},
        'left': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
        'top': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
        'width': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
        'height': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
        'color': {'is_sequence': True, 'type': 'categorical', 'input_dim': 16, 'shape': [3]},
        'image_embedding': {'is_sequence': True, 'type': 'numerical', 'shape': [512]},
        'text_embedding': {'is_sequence': True, 'type': 'numerical', 'shape': [512]},
    }
    
    # 创建模型
    print("创建模型...")
    model = MFP(
        input_columns,
        embed_dim=args.embed_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    
    # 加载预训练权重
    if args.pretrained:
        print(f"加载预训练权重: {args.pretrained}")
        model.load_converted_weights(args.pretrained)
    
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = MFPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
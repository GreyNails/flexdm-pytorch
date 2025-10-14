"""
PyTorch版本的Demo/推理代码
对应原始demo_crello.ipynb的功能
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Optional
import argparse

from models_pytorch import MFP
from dataset import DesignLayoutDataset
from masking_pytorch import (
    preprocess_for_test,
    merge_inputs_and_prediction,
    get_initial_masks,
    get_seq_mask,
    get_attribute_groups,
)
from svg_builder_pytorch import SVGBuilder
from retriever_pytorch import ImageRetriever, TextRetriever


class MFPDemo:
    """MFP模型演示器"""
    
    def __init__(
        self,
        model: MFP,
        dataset: DesignLayoutDataset,
        image_retriever: Optional[ImageRetriever] = None,
        text_retriever: Optional[TextRetriever] = None,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.model.eval()
        self.dataset = dataset
        self.device = device
        self.input_columns = model.input_columns
        
        # 构建SVG生成器
        self.builders = self._create_builders(image_retriever, text_retriever)
    
    def _create_builders(self, image_db, text_db):
        """创建SVG构建器"""
        builders = {}
        
        # Layout视图（显示类型）
        builders['layout'] = SVGBuilder(
            key='type',
            preprocessor={'type': {'vocabulary': list(self.dataset.idx_to_type.values())}},
            max_width=128,
            max_height=192,
        )
        
        # Visual视图（显示颜色+内容）
        builders['visual'] = SVGBuilder(
            key='color',
            max_width=128,
            max_height=192,
            image_db=image_db,
            text_db=text_db,
            render_text=True,
        )
        
        # Visual without text
        builders['visual_wo_text'] = SVGBuilder(
            key='color',
            max_width=128,
            max_height=192,
            image_db=image_db,
            text_db=None,
            render_text=False,
        )
        
        # Visual without image
        builders['visual_wo_image'] = SVGBuilder(
            key='color',
            max_width=128,
            max_height=192,
            image_db=None,
            text_db=text_db,
            render_text=True,
        )
        
        return builders
    
    def unbatch(self, batch: Dict, batch_idx: int = 0) -> Dict:
        """
        将batch转换为单个样本（用于可视化）
        
        Args:
            batch: 批次数据
            batch_idx: 要提取的样本索引
        
        Returns:
            单个样本的字典
        """
        sample = {}
        
        # ID
        if isinstance(batch['id'], list):
            sample['id'] = batch['id'][batch_idx]
        else:
            sample['id'] = 'sample'
        
        # Canvas信息
        canvas_width_idx = batch['canvas_width'][batch_idx, 0].item()
        canvas_height_idx = batch['canvas_height'][batch_idx, 0].item()
        sample['canvas_width'] = self.dataset.idx_to_width.get(canvas_width_idx, 800)
        sample['canvas_height'] = self.dataset.idx_to_height.get(canvas_height_idx, 600)
        
        # Length
        length = batch['length'][batch_idx, 0].item() + 1  # zero-based to one-based
        
        # 元素列表
        elements = []
        for i in range(length):
            elem = {}
            
            # Type
            type_idx = batch['type'][batch_idx, i, 0].item()
            elem['type'] = self.dataset.idx_to_type.get(type_idx, 'unknown')
            
            # Position (de-discretize)
            elem['left'] = batch['left'][batch_idx, i, 0].item() / (self.dataset.bins - 1)
            elem['top'] = batch['top'][batch_idx, i, 0].item() / (self.dataset.bins - 1)
            elem['width'] = batch['width'][batch_idx, i, 0].item() / (self.dataset.bins - 1)
            elem['height'] = batch['height'][batch_idx, i, 0].item() / (self.dataset.bins - 1)
            
            # Opacity
            if 'opacity' in batch:
                elem['opacity'] = batch['opacity'][batch_idx, i, 0].item() / 7.0
            
            # Color
            if 'color' in batch:
                color = batch['color'][batch_idx, i].cpu().numpy()
                elem['color'] = [int(c * 255 / 15) for c in color]
            
            # Font
            if 'font_family' in batch:
                font_idx = batch['font_family'][batch_idx, i, 0].item()
                elem['font_family'] = self.dataset.idx_to_font.get(font_idx, 'Arial')
            
            # Embeddings
            if 'image_embedding' in batch:
                elem['image_embedding'] = batch['image_embedding'][batch_idx, i].cpu().numpy()
            if 'text_embedding' in batch:
                elem['text_embedding'] = batch['text_embedding'][batch_idx, i].cpu().numpy()
            
            # UUID
            if 'uuid' in batch and isinstance(batch['uuid'], list):
                elem['uuid'] = batch['uuid'][batch_idx][i] if i < len(batch['uuid'][batch_idx]) else ''
            
            elements.append(elem)
        
        sample['elements'] = elements
        return sample
    
    @torch.no_grad()
    def predict(
        self,
        batch: Dict,
        task: str = 'pos',
        mask_positions: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        运行预测
        
        Args:
            batch: 输入批次
            task: 任务类型 ('pos', 'elem', 'attr', 'txt', 'img')
            mask_positions: 自定义mask位置（可选）
        
        Returns:
            预测结果
        """
        # 移动到设备
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v 
                 for k, v in batch.items()}
        
        # 生成mask
        seq_mask = get_seq_mask(inputs['length'], max_len=inputs['left'].size(1))
        mfp_masks = get_initial_masks(self.input_columns, seq_mask)
        
        # 根据任务类型设置mask
        if mask_positions is not None:
            # 使用自定义mask
            for key in mfp_masks.keys():
                if self.input_columns[key].get('is_sequence', False):
                    mfp_masks[key] = mask_positions
        else:
            # 根据任务类型设置mask
            attr_groups = get_attribute_groups(self.input_columns.keys())
            
            if task == 'elem':
                # Mask第一个元素
                for key in mfp_masks.keys():
                    if self.input_columns[key].get('is_sequence', False):
                        mfp_masks[key][:, 0] = True
            else:
                # Mask特定属性组
                if task in attr_groups:
                    feat_group = attr_groups[task]
                    for key in feat_group:
                        if key in mfp_masks:
                            mfp_masks[key] = seq_mask
        
        # 预处理
        modified_inputs = preprocess_for_test(inputs, self.input_columns, mfp_masks)
        
        # 前向传播
        outputs = self.model(modified_inputs)
        
        # 合并结果
        merged = merge_inputs_and_prediction(inputs, self.input_columns, mfp_masks, outputs)
        
        # 将分类预测转换为类别索引
        for key, column in self.input_columns.items():
            if column.get('type') == 'categorical' and key in merged:
                if merged[key].dim() == 4:  # (B, S, num_feat, C)
                    merged[key] = merged[key].argmax(dim=-1)
                elif merged[key].dim() == 3:  # (B, S, C)
                    merged[key] = merged[key].argmax(dim=-1).unsqueeze(-1)
        
        return merged
    
    def visualize(
        self,
        batch: Dict,
        task: str = 'pos',
        save_dir: Optional[Path] = None,
    ) -> List[str]:
        """
        可视化结果
        
        Args:
            batch: 输入批次
            task: 任务类型
            save_dir: 保存目录（可选）
        
        Returns:
            SVG字符串列表
        """
        # 生成预测
        predictions = self.predict(batch, task=task)
        
        # 转换为CPU
        batch_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in batch.items()}
        pred_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in predictions.items()}
        
        # 生成SVG
        batch_size = batch['length'].size(0)
        svgs = []
        
        for i in range(batch_size):
            row_svgs = []
            
            # Ground truth - layout
            gt_doc = self.unbatch(batch_cpu, i)
            row_svgs.append(self.builders['layout'](gt_doc))
            
            # Ground truth - visual
            row_svgs.append(self.builders['visual'](gt_doc))
            
            # Prediction
            if task in ['pos', 'elem']:
                # 显示layout
                pred_doc = self.unbatch(pred_cpu, i)
                row_svgs.append(self.builders['layout'](pred_doc))
            
            # Prediction - visual
            pred_doc = self.unbatch(pred_cpu, i)
            row_svgs.append(self.builders['visual'](pred_doc))
            
            svgs.append(row_svgs)
            
            # 保存到文件
            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                
                for j, svg in enumerate(row_svgs):
                    filename = save_dir / f'sample_{i}_view_{j}.svg'
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(svg)
        
        return svgs
    
    def demo(
        self,
        num_samples: int = 5,
        task: str = 'pos',
        save_dir: Optional[str] = None,
    ):
        """
        运行完整demo
        
        Args:
            num_samples: 样本数量
            task: 任务类型
            save_dir: 保存目录
        """
        print(f"\n{'='*60}")
        print(f"MFP Demo - Task: {task}")
        print(f"{'='*60}\n")
        
        # 获取样本
        indices = list(range(min(num_samples, len(self.dataset))))
        samples = [self.dataset[i] for i in indices]
        
        # 构建batch
        from torch.utils.data import default_collate
        batch = default_collate(samples)
        
        # 可视化
        save_path = Path(save_dir) if save_dir else None
        svgs = self.visualize(batch, task=task, save_dir=save_path)
        
        print(f"✓ 生成了 {len(svgs)} 个样本的可视化")
        if save_path:
            print(f"✓ 保存到: {save_path}")
        
        return svgs


def main():
    parser = argparse.ArgumentParser(description='MFP Demo')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据目录')
    parser.add_argument('--input_columns', type=str, required=True,
                       help='input_columns JSON文件')
    parser.add_argument('--task', type=str, default='pos',
                       choices=['pos', 'elem', 'attr', 'txt', 'img'],
                       help='任务类型')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='样本数量')
    parser.add_argument('--save_dir', type=str, default='./demo_output',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--build_retriever', action='store_true',
                       help='是否构建检索器（需要时间）')
    
    args = parser.parse_args()
    
    # 加载input_columns
    print("加载input_columns...")
    with open(args.input_columns, 'r') as f:
        input_columns = json.load(f)
    
    # 创建模型
    print("创建模型...")
    model = MFP(
        input_columns=input_columns,
        embed_dim=256,
        num_blocks=4,
        num_heads=8,
        dropout=0.1,
        max_length=50,
    )
    
    # 加载权重
    print(f"加载checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建数据集
    print("加载数据集...")
    dataset = DesignLayoutDataset(
        args.data_dir,
        split='test',
        max_length=20,
    )
    
    # 构建检索器（可选）
    image_retriever = None
    text_retriever = None
    
    if args.build_retriever:
        print("构建检索器...")
        data_path = Path(args.data_dir)
        
        image_retriever = ImageRetriever(
            data_path,
            image_path=data_path.parent / "crello" / "images"
        )
        image_retriever.build("test")
        
        text_retriever = TextRetriever(
            data_path,
            text_path=data_path.parent / "crello" / "texts"
        )
        text_retriever.build("test")
    
    # 创建Demo
    demo = MFPDemo(
        model=model,
        dataset=dataset,
        image_retriever=image_retriever,
        text_retriever=text_retriever,
        device=args.device,
    )
    
    # 运行demo
    demo.demo(
        num_samples=args.num_samples,
        task=args.task,
        save_dir=args.save_dir,
    )
    
    print("\n✓ Demo完成!")


if __name__ == "__main__":
    main()
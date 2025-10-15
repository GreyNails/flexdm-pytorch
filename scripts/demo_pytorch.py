"""
MFP PyTorch Demo - 改进版
支持CPU/CUDA，整合完整的Masking机制
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import torch
import numpy as np

# 设置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入项目模块
from models_pytorch import MFP
from dataset import DesignLayoutDataset, collate_fn
from masking_pytorch import (
    get_seq_mask,
    get_initial_masks,
    preprocess_for_test,
    merge_inputs_and_prediction,
    get_attribute_groups,
)
from svg_builder_pytorch import SVGBuilder
from retriever_pytorch import ImageRetriever, TextRetriever


class MFPDemoRunner:
    """MFP Demo运行器 - 改进版"""
    
    def __init__(
        self,
        checkpoint_path: str,
        data_dir: str,
        input_columns_path: str,
        device: str = 'cpu',
        batch_size: int = 4,
        build_retriever: bool = False,
    ):
        """
        初始化Demo运行器
        
        Args:
            checkpoint_path: 模型checkpoint路径
            data_dir: 数据目录
            input_columns_path: input_columns配置文件
            device: 运行设备 ('cpu' or 'cuda')
            batch_size: 批次大小
            build_retriever: 是否构建图像/文本检索器
        """
        self.device = device
        self.batch_size = batch_size
        self.data_dir = Path(data_dir)
        
        logger.info("="*80)
        logger.info("初始化MFP Demo")
        logger.info("="*80)
        
        # 检查CUDA
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA不可用，切换到CPU")
            self.device = 'cpu'
        
        logger.info(f"设备: {self.device}")
        logger.info(f"批次大小: {batch_size}")
        
        # 1. 加载input_columns
        logger.info("\n1. 加载配置")
        with open(input_columns_path, 'r') as f:
            self.input_columns = json.load(f)
        logger.info(f"✓ Input columns: {len(self.input_columns)} 个特征")
        
        # 2. 加载数据集
        logger.info("\n2. 加载数据集")
        self.dataset = DesignLayoutDataset(
            str(self.data_dir),
            split='test',
            max_length=20,
        )
        logger.info(f"✓ 数据集大小: {len(self.dataset)} 个样本")
        
        # 3. 创建映射
        self._create_mappings()
        
        # 4. 加载模型
        logger.info("\n4. 加载模型")
        self.model = self._load_model(checkpoint_path)
        
        # 5. 构建检索器（可选）
        self.image_db = None
        self.text_db = None
        if build_retriever:
            logger.info("\n5. 构建检索器")
            self._build_retrievers()
        else:
            logger.info("\n5. 跳过检索器构建")
        
        # 6. 创建SVG构建器
        logger.info("\n6. 创建SVG构建器")
        self.builders = self._create_builders()
        
        # 获取属性组
        self.attribute_groups = get_attribute_groups(self.input_columns.keys())
        
        logger.info("\n" + "="*80)
        logger.info("✓ 初始化完成")
        logger.info("="*80 + "\n")
    
    def _create_mappings(self):
        """创建类型、字体、画布尺寸映射"""
        logger.info("\n3. 创建映射")
        
        # 类型映射
        self.type_mapping = self.dataset.idx_to_type
        logger.info(f"✓ 类型映射: {len(self.type_mapping)} 个类型")
        
        # 字体映射
        self.font_mapping = getattr(self.dataset, 'idx_to_font', {})
        if self.font_mapping:
            logger.info(f"✓ 字体映射: {len(self.font_mapping)} 个字体")
        
        # 画布尺寸映射
        self.width_mapping = getattr(self.dataset, 'idx_to_width', {0: 800})
        self.height_mapping = getattr(self.dataset, 'idx_to_height', {0: 600})
        logger.info(f"✓ 画布尺寸映射: {len(self.width_mapping)} x {len(self.height_mapping)}")
    
    def _load_model(self, checkpoint_path: str):
        """加载模型"""
        model = MFP(
            input_columns=self.input_columns,
            embed_dim=256,
            num_blocks=4,
            num_heads=8,
            dropout=0.1,
            max_length=50,
        )
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        if missing:
            logger.warning(f"缺失 {len(missing)} 个键: {missing[:3]}...")
        if unexpected:
            logger.warning(f"多余 {len(unexpected)} 个键: {unexpected[:3]}...")
        
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"✓ 模型加载完成: {checkpoint_path}")
        return model
    
    def _build_retrievers(self):
        """构建图像和文本检索器"""
        db_root = self.data_dir.parent / "crello"
        
        # 图像检索器
        self.image_db = ImageRetriever(
            self.data_dir,
            image_path=db_root / "images"
        )
        try:
            self.image_db.build("test")
            logger.info("✓ 图像检索器构建完成")
        except Exception as e:
            logger.warning(f"图像检索器构建失败: {e}")
            self.image_db = None
        
        # 文本检索器
        self.text_db = TextRetriever(
            self.data_dir,
            text_path=db_root / "texts"
        )
        try:
            self.text_db.build("test")
            logger.info("✓ 文本检索器构建完成")
        except Exception as e:
            logger.warning(f"文本检索器构建失败: {e}")
            self.text_db = None
    
    def _create_builders(self):
        """创建SVG构建器"""
        builders = {}
        
        # Layout视图
        builders['layout'] = SVGBuilder(
            key='type',
            preprocessor={'type': {'vocabulary': list(self.type_mapping.values())}},
            max_width=128,
            max_height=192,
            opacity=0.8,
        )
        
        # Visual视图
        builders['visual'] = SVGBuilder(
            key='color',
            max_width=128,
            max_height=192,
            image_db=self.image_db,
            text_db=self.text_db,
            render_text=True,
            opacity=1.0,
        )
        
        # Visual without text
        builders['visual_wo_text'] = SVGBuilder(
            key='color',
            max_width=128,
            max_height=192,
            image_db=self.image_db,
            text_db=None,
            render_text=False,
            opacity=1.0,
        )
        
        # Visual without image
        builders['visual_wo_image'] = SVGBuilder(
            key='color',
            max_width=128,
            max_height=192,
            image_db=None,
            text_db=self.text_db,
            render_text=True,
            opacity=1.0,
        )
        
        logger.info("✓ SVG构建器创建完成")
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
        if 'id' in batch:
            sample['id'] = batch['id'][batch_idx] if isinstance(batch['id'], list) else 'sample'
        else:
            sample['id'] = 'sample'
        
        # Canvas信息
        canvas_width_idx = batch['canvas_width'][batch_idx, 0].item()
        canvas_height_idx = batch['canvas_height'][batch_idx, 0].item()
        sample['canvas_width'] = self.width_mapping.get(canvas_width_idx, 800)
        sample['canvas_height'] = self.height_mapping.get(canvas_height_idx, 600)
        
        # Length
        length = batch['length'][batch_idx, 0].item() + 1  # zero-based to one-based
        
        # 元素列表
        elements = []
        for i in range(length):
            elem = {}
            
            # Type
            type_val = batch['type'][batch_idx, i, 0]
            if type_val.dim() == 1:  # (C,) logits
                type_idx = type_val.argmax().item()
            else:
                type_idx = type_val.item()
            elem['type'] = self.type_mapping.get(type_idx, 'unknown')
            
            # Position (de-discretize)
            for pos_key in ['left', 'top', 'width', 'height']:
                pos_val = batch[pos_key][batch_idx, i, 0]
                if pos_val.dim() == 1:  # logits
                    pos_idx = pos_val.argmax().item()
                else:
                    pos_idx = pos_val.item()
                elem[pos_key] = pos_idx / (self.dataset.bins - 1)
            
            # Opacity
            if 'opacity' in batch:
                opacity_val = batch['opacity'][batch_idx, i, 0]
                if opacity_val.dim() == 1:
                    opacity_idx = opacity_val.argmax().item()
                else:
                    opacity_idx = opacity_val.item()
                elem['opacity'] = opacity_idx / 7.0
            
            # Color
            if 'color' in batch:
                color_val = batch['color'][batch_idx, i]  # (3, C) or (3,)
                if color_val.dim() == 2:  # (3, C) logits
                    color_indices = color_val.argmax(dim=-1).cpu().numpy()
                else:  # (3,)
                    color_indices = color_val.cpu().numpy()
                elem['color'] = [int(c * 255 / 15) for c in color_indices]
            
            # Font
            if 'font_family' in batch:
                font_val = batch['font_family'][batch_idx, i, 0]
                if font_val.dim() == 1:
                    font_idx = font_val.argmax().item()
                else:
                    font_idx = font_val.item()
                elem['font_family'] = self.font_mapping.get(font_idx, 'Arial')
            
            # Embeddings
            if 'image_embedding' in batch:
                elem['image_embedding'] = batch['image_embedding'][batch_idx, i].cpu().numpy()
            if 'text_embedding' in batch:
                elem['text_embedding'] = batch['text_embedding'][batch_idx, i].cpu().numpy()
            
            # UUID
            if 'uuid' in batch and isinstance(batch['uuid'], list):
                if i < len(batch['uuid'][batch_idx]):
                    elem['uuid'] = batch['uuid'][batch_idx][i]
            
            elements.append(elem)
        
        sample['elements'] = elements
        return sample
    
    @torch.no_grad()
    def run_inference(
        self,
        batch: Dict,
        task: str = 'pos',
    ) -> Dict:
        """
        运行推理
        
        Args:
            batch: 输入批次
            task: 任务类型 ('pos', 'elem', 'attr', 'txt', 'img')
        
        Returns:
            预测结果
        """
        # 移动到设备
        inputs = {
            k: v.to(self.device) if torch.is_tensor(v) else v 
            for k, v in batch.items()
        }
        
        # 生成mask
        seq_mask = get_seq_mask(inputs['length'], max_len=inputs['left'].size(1))
        mfp_masks = get_initial_masks(self.input_columns, seq_mask)
        
        # 根据任务类型设置mask
        if task == 'elem':
            # Mask第一个元素的所有特征
            for key in mfp_masks.keys():
                if self.input_columns[key].get('is_sequence', False):
                    mfp_masks[key][:, 0] = True
        else:
            # Mask特定属性组
            if task in self.attribute_groups:
                feat_group = self.attribute_groups[task]
                for key in feat_group:
                    if key in mfp_masks and key != 'type':
                        mfp_masks[key] = seq_mask.clone()
        
        # 预处理（应用masking）
        modified_inputs = preprocess_for_test(inputs, self.input_columns, mfp_masks)
        
        # 前向传播
        outputs = self.model(modified_inputs)
        
        # 合并结果（未mask的部分保持原值）
        merged = merge_inputs_and_prediction(inputs, self.input_columns, mfp_masks, outputs)
        
        return merged
    
    def visualize(
        self,
        batch: Dict,
        task: str = 'pos',
        save_dir: Optional[Path] = None,
    ) -> List[List[str]]:
        """
        可视化结果
        
        Args:
            batch: 输入批次
            task: 任务类型
            save_dir: 保存目录（可选）
        
        Returns:
            SVG字符串列表
        """
        logger.info(f"\n开始可视化 - 任务: {task}")
        
        # 确保数据在正确设备
        batch = {
            k: v.to(self.device) if torch.is_tensor(v) else v 
            for k, v in batch.items()
        }
        
        batch_size = batch['length'].size(0)
        svgs = []
        
        # Ground Truth
        logger.info("  - 渲染 GT Layout")
        gt_layout = []
        for i in range(batch_size):
            gt_doc = self.unbatch(batch, i)
            gt_layout.append(self.builders['layout'](gt_doc))
        svgs.append(gt_layout)
        
        logger.info("  - 渲染 GT Visual")
        gt_visual = []
        for i in range(batch_size):
            gt_doc = self.unbatch(batch, i)
            gt_visual.append(self.builders['visual'](gt_doc))
        svgs.append(gt_visual)
        
        # 输入视图（根据任务类型）
        if task == 'txt':
            logger.info("  - 渲染 Input (无文本)")
            input_visual = []
            for i in range(batch_size):
                doc = self.unbatch(batch, i)
                input_visual.append(self.builders['visual_wo_text'](doc))
            svgs.append(input_visual)
        elif task == 'img':
            logger.info("  - 渲染 Input (无图像)")
            input_visual = []
            for i in range(batch_size):
                doc = self.unbatch(batch, i)
                input_visual.append(self.builders['visual_wo_image'](doc))
            svgs.append(input_visual)
        elif task == 'attr':
            logger.info("  - 渲染 Input (默认属性)")
            input_visual = []
            for i in range(batch_size):
                doc = self.unbatch(batch, i)
                # 设置默认视觉属性
                for elem in doc['elements']:
                    elem.setdefault('color', [128, 128, 128])
                    elem.setdefault('opacity', 1.0)
                    elem.setdefault('font_family', 'Arial')
                input_visual.append(self.builders['visual'](doc))
            svgs.append(input_visual)
        
        # 元素级任务：显示输入（去掉被mask的元素）
        if task == 'elem':
            logger.info("  - 渲染 Input (缺少元素)")
            # 创建副本，去掉第一个元素
            batch_copy = {}
            for key, value in batch.items():
                if torch.is_tensor(value) and value.dim() >= 2 and value.size(1) > 1:
                    # 去掉第一个元素
                    batch_copy[key] = value[:, 1:]
                else:
                    batch_copy[key] = value
            
            batch_copy['length'] = batch['length'] - 1
            
            input_layout = []
            input_visual = []
            for i in range(batch_size):
                doc = self.unbatch(batch_copy, i)
                input_layout.append(self.builders['layout'](doc))
                input_visual.append(self.builders['visual'](doc))
            svgs.append(input_layout)
            svgs.append(input_visual)
        
        # 预测
        logger.info("  - 运行模型推理")
        predictions = self.run_inference(batch, task=task)
        
        # 转到CPU
        predictions = {
            k: v.cpu() if torch.is_tensor(v) else v 
            for k, v in predictions.items()
        }
        
        # 渲染预测结果
        if task in ['pos', 'elem']:
            logger.info("  - 渲染 Pred Layout")
            pred_layout = []
            for i in range(batch_size):
                pred_doc = self.unbatch(predictions, i)
                pred_layout.append(self.builders['layout'](pred_doc))
            svgs.append(pred_layout)
        
        logger.info("  - 渲染 Pred Visual")
        pred_visual = []
        for i in range(batch_size):
            pred_doc = self.unbatch(predictions, i)
            pred_visual.append(self.builders['visual'](pred_doc))
        svgs.append(pred_visual)
        
        # 转置（每个样本一行）
        svgs_transposed = list(zip(*svgs))
        
        # 保存到文件（可选）
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"\n保存SVG到: {save_dir}")
            for i, row in enumerate(svgs_transposed):
                for j, svg in enumerate(row):
                    filename = save_dir / f'sample_{i}_view_{j}.svg'
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(svg)
            
            # 创建HTML索引
            self._create_html_index(save_dir, len(svgs_transposed), len(row), task)
        
        logger.info("✓ 可视化完成\n")
        return svgs_transposed
    
    def _create_html_index(self, save_dir: Path, num_samples: int, num_views: int, task: str):
        """创建HTML索引页面"""
        html_path = save_dir / 'index.html'
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MFP Demo Results - {task}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        .sample {{
            background: white;
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .views {{
            display: flex;
            gap: 10px;
            overflow-x: auto;
        }}
        .view {{
            flex-shrink: 0;
        }}
        .view img {{
            width: 200px;
            height: auto;
            border: 1px solid #ddd;
        }}
        .view-label {{
            text-align: center;
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <h1>MFP Demo Results - Task: {task}</h1>
    <p>Total: {num_samples} samples</p>
""")
            
            view_labels = self._get_view_labels(task)
            
            for i in range(num_samples):
                f.write(f'    <div class="sample">\n')
                f.write(f'        <h2>Sample {i}</h2>\n')
                f.write(f'        <div class="views">\n')
                
                for j in range(num_views):
                    label = view_labels[j] if j < len(view_labels) else f'View {j}'
                    f.write(f'            <div class="view">\n')
                    f.write(f'                <img src="sample_{i}_view_{j}.svg" alt="{label}">\n')
                    f.write(f'                <div class="view-label">{label}</div>\n')
                    f.write(f'            </div>\n')
                
                f.write(f'        </div>\n')
                f.write(f'    </div>\n')
            
            f.write("""</body>
</html>
""")
        
        logger.info(f"✓ HTML索引: {html_path}")
    
    def _get_view_labels(self, task: str) -> List[str]:
        """获取视图标签"""
        labels = {
            'pos': ['GT Layout', 'GT Visual', 'Pred Layout', 'Pred Visual'],
            'elem': ['GT Layout', 'GT Visual', 'Input Layout', 'Input Visual', 
                    'Pred Layout', 'Pred Visual'],
            'attr': ['GT Layout', 'GT Visual', 'Input', 'Pred Visual'],
            'txt': ['GT Layout', 'GT Visual', 'Input', 'Pred Visual'],
            'img': ['GT Layout', 'GT Visual', 'Input', 'Pred Visual'],
        }
        return labels.get(task, ['GT Layout', 'GT Visual', 'Pred'])
    
    def run_demo(
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
        logger.info("="*80)
        logger.info(f"运行Demo - 任务: {task}, 样本数: {num_samples}")
        logger.info("="*80)
        
        # 创建DataLoader
        from torch.utils.data import DataLoader, Subset
        
        indices = list(range(min(num_samples, len(self.dataset))))
        subset = Subset(self.dataset, indices)
        dataloader = DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        all_svgs = []
        for batch_idx, batch in enumerate(dataloader):
            logger.info(f"\n处理批次 {batch_idx + 1}/{len(dataloader)}")
            svgs = self.visualize(batch, task=task, save_dir=None)
            all_svgs.extend(svgs)
        
        # 保存所有结果
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"\n保存所有结果到: {save_path}")
            for i, row in enumerate(all_svgs):
                for j, svg in enumerate(row):
                    filename = save_path / f'sample_{i}_view_{j}.svg'
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(svg)
            
            # 创建HTML索引
            self._create_html_index(save_path, len(all_svgs), len(all_svgs[0]), task)
            
            logger.info(f"\n✓ 结果已保存")
            logger.info(f"  SVG文件: {len(all_svgs)} x {len(all_svgs[0])} = {len(all_svgs) * len(all_svgs[0])} 个")
            logger.info(f"  HTML索引: {save_path / 'index.html'}")
        
        logger.info("\n" + "="*80)
        logger.info("✓ Demo完成!")
        logger.info("="*80 + "\n")
        
        return all_svgs


def main():
    parser = argparse.ArgumentParser(description='MFP Demo - 改进版')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据目录')
    parser.add_argument('--input_columns', type=str, required=True,
                       help='input_columns配置文件')
    
    # 可选参数
    parser.add_argument('--task', type=str, default='pos',
                       choices=['pos', 'elem', 'attr', 'txt', 'img'],
                       help='任务类型')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='样本数量')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cpu', 'cuda'],
                       help='运行设备')
    parser.add_argument('--save_dir', type=str, default='./demo_output',
                       help='输出目录')
    parser.add_argument('--build_retriever', action='store_true',
                       help='是否构建检索器（需要时间）')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(0)
    np.random.seed(0)
    
    # 创建Demo运行器
    demo = MFPDemoRunner(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        input_columns_path=args.input_columns,
        device=args.device,
        batch_size=args.batch_size,
        build_retriever=args.build_retriever,
    )
    
    # 运行Demo
    demo.run_demo(
        num_samples=args.num_samples,
        task=args.task,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
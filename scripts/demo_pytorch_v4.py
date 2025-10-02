"""
PyTorch版本的MFP模型演示和可视化
支持将SVG结果保存为PNG图片
"""
import json
import itertools
import logging
import io
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# SVG转PNG依赖
try:
    import cairosvg
    HAS_CAIROSVG = True
except ImportError:
    HAS_CAIROSVG = False
    print("警告: 未安装cairosvg，将使用备用方案")
    print("安装: pip install cairosvg")

try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    HAS_SVGLIB = True
except ImportError:
    HAS_SVGLIB = False

# 导入PyTorch模型和工具
from models_pytorch import MFP
from dataset import DesignLayoutDataset
from svg_builder_pytorch import SVGBuilder
from retriever_pytorch import ImageRetriever, TextRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)


class DemoConfig:
    """演示配置"""
    def __init__(self):
        self.ckpt_dir = "/home/dell/Project-HCL/BaseLine/flexdm_pt/chechpoints"
        self.dataset_name = "crello_json"
        self.db_root = "/home/dell/Project-HCL/BaseLine/flexdm_pt/data/crello_json"
        self.batch_size = 20
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 任务类型: elem, pos, attr, txt, img
        self.target_task = "pos"
        
        # 输出目录
        self.output_dir = "./outputs"
        
        # 列名配置
        self.column_names = {
            "txt": ["gt-layout", "gt-visual", "input", "pred"],
            "img": ["gt-layout", "gt-visual", "input", "pred"],
            "attr": ["gt-layout", "gt-visual", "input", "pred"],
            "pos": ["gt-layout", "gt-visual", "pred-layout", "pred-visual"],
            "elem": ["gt-layout", "gt-visual", "input-layout", "input-visual", "pred-layout", "pred-visual"],
        }
        
        # 属性分组
        self.attribute_groups = {
            "type": ["type"],
            "pos": ["left", "top", "width", "height"],
            "attr": ["opacity", "color", "font_family"],
            "img": ["image_embedding"],
            "txt": ["text_embedding"],
        }


# ==================== SVG保存功能 ====================

def save_svg_as_image(
    svg_string: str,
    output_path: str,
    width: int = 800,
    height: int = 600
):
    """
    将SVG字符串保存为PNG图片
    
    Args:
        svg_string: SVG内容字符串
        output_path: 输出文件路径
        width: 图片宽度
        height: 图片高度
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if HAS_CAIROSVG:
        # 方案1: cairosvg (最佳质量)
        cairosvg.svg2png(
            bytestring=svg_string.encode('utf-8'),
            write_to=str(output_path),
            output_width=width,
            output_height=height,
        )
    elif HAS_SVGLIB:
        # 方案2: svglib (备用)
        try:
            drawing = svg2rlg(io.BytesIO(svg_string.encode('utf-8')))
            renderPM.drawToFile(drawing, str(output_path), fmt='PNG')
        except Exception as e:
            logger.warning(f"svglib转换失败: {e}")
            # 降级到SVG
            svg_path = output_path.with_suffix('.svg')
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(svg_string)
    else:
        # 方案3: 直接保存SVG文件
        svg_path = output_path.with_suffix('.svg')
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_string)
        logger.warning(f"保存为SVG格式: {svg_path}")


def save_visualization_grid(
    svgs: list,
    output_dir: str,
    column_names: list,
    prefix: str = "sample"
):
    """
    保存可视化结果为网格图片
    
    Args:
        svgs: SVG列表 [[row1_cols], [row2_cols], ...]
        output_dir: 输出目录
        column_names: 列名列表
        prefix: 文件名前缀
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n保存可视化结果到: {output_dir}")
    
    # 保存每个样本的所有列
    for sample_idx, row in enumerate(svgs):
        sample_dir = output_dir / f"{prefix}_{sample_idx:03d}"
        sample_dir.mkdir(exist_ok=True)
        
        logger.info(f"  样本 {sample_idx}:")
        
        # 展平嵌套列表
        flat_row = []
        for item in row:
            if isinstance(item, (list, tuple)):
                flat_row.extend(item)
            else:
                flat_row.append(item)
        
        # 保存每一列
        saved_count = 0
        for col_idx, col_name in enumerate(column_names):
            if col_idx >= len(flat_row):
                break
            
            svg_content = flat_row[col_idx]
            if svg_content is None:
                continue
            
            output_path = sample_dir / f"{col_idx:02d}_{col_name}.png"
            try:
                save_svg_as_image(svg_content, str(output_path), width=400, height=600)
                saved_count += 1
            except Exception as e:
                logger.error(f"    ✗ {col_name}: {e}")
        
        logger.info(f"    ✓ 保存了 {saved_count}/{len(column_names)} 张图片")
    
    logger.info(f"\n✓ 所有结果已保存到: {output_dir}")


def create_comparison_image(
    svgs: list,
    output_path: str,
    column_names: list,
    samples_per_row: int = 2,
    svg_width: int = 200,
    svg_height: int = 300,
):
    """
    创建对比图（多个样本并排显示）
    
    Args:
        svgs: SVG列表
        output_path: 输出文件路径
        column_names: 列名
        samples_per_row: 每行显示的样本数
        svg_width: 单个SVG宽度
        svg_height: 单个SVG高度
    """
    if not HAS_CAIROSVG:
        logger.warning("需要安装cairosvg才能创建对比图: pip install cairosvg")
        return
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    padding = 10
    header_height = 30
    label_height = 20
    
    num_samples = len(svgs)
    num_cols = len(column_names)
    num_rows = (num_samples + samples_per_row - 1) // samples_per_row
    
    # 计算总画布大小
    cell_width = svg_width + padding * 2
    cell_height = svg_height + header_height + label_height + padding * 2
    canvas_width = samples_per_row * num_cols * cell_width
    canvas_height = num_rows * cell_height
    
    # 创建画布
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # 加载字体
    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
        label_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
        )
    except:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # 处理每个样本
    for sample_idx, row in enumerate(svgs):
        row_idx = sample_idx // samples_per_row
        col_idx = sample_idx % samples_per_row
        
        base_x = col_idx * num_cols * cell_width
        base_y = row_idx * cell_height
        
        # 展平行
        flat_row = []
        for item in row:
            if isinstance(item, (list, tuple)):
                flat_row.extend(item)
            else:
                flat_row.append(item)
        
        # 渲染每一列
        for c_idx, col_name in enumerate(column_names):
            if c_idx >= len(flat_row):
                break
            
            svg_content = flat_row[c_idx]
            if svg_content is None:
                continue
            
            x = base_x + c_idx * cell_width + padding
            y = base_y + header_height + padding
            
            try:
                # 添加列标题（第一行样本）
                if row_idx == 0:
                    draw.text(
                        (x + svg_width // 2, base_y + 5),
                        col_name,
                        fill='black',
                        font=title_font,
                        anchor="mt"
                    )
                
                # 转换SVG为PNG
                png_data = cairosvg.svg2png(
                    bytestring=svg_content.encode('utf-8'),
                    output_width=svg_width,
                    output_height=svg_height,
                )
                img = Image.open(io.BytesIO(png_data))
                canvas.paste(img, (x, y))
                
                # 添加样本标签（第一列）
                if c_idx == 0:
                    draw.text(
                        (x - 5, y + svg_height // 2),
                        f"#{sample_idx}",
                        fill='black',
                        font=label_font,
                        anchor="rm"
                    )
                
            except Exception as e:
                logger.error(f"渲染失败: Sample {sample_idx}, {col_name}: {e}")
                # 绘制错误占位符
                draw.rectangle(
                    [x, y, x + svg_width, y + svg_height],
                    outline='red',
                    width=2
                )
                draw.text(
                    (x + svg_width // 2, y + svg_height // 2),
                    "Error",
                    fill='red',
                    font=label_font,
                    anchor="mm"
                )
    
    # 保存
    canvas.save(output_path)
    logger.info(f"✓ 对比图已保存: {output_path}")


# ==================== 模型和数据加载 ====================

def load_model(checkpoint_path: str, input_columns: Dict, device: str = 'cuda'):
    """加载PyTorch模型"""
    model = MFP(
        input_columns=input_columns,
        embed_dim=256,
        num_blocks=4,
        num_heads=8,
        dropout=0.1,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        logger.warning(f"Missing keys: {len(missing)}")
    if unexpected:
        logger.warning(f"Unexpected keys: {len(unexpected)}")
    
    model.to(device)
    model.eval()
    
    logger.info(f"✓ Model loaded from {checkpoint_path}")
    return model


def get_seq_mask(lengths: torch.Tensor, max_len: int = None) -> torch.Tensor:
    """生成序列掩码"""
    if lengths.dim() == 2:
        lengths = lengths.squeeze(-1)
    
    batch_size = lengths.size(0)
    if max_len is None:
        max_len = lengths.max().item()
    
    mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    return mask


def get_initial_masks(input_columns: Dict, seq_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """初始化掩码字典"""
    masks = {}
    batch_size, seq_len = seq_mask.shape
    
    for key, column in input_columns.items():
        if column.get('is_sequence', False):
            masks[key] = torch.zeros_like(seq_mask, dtype=torch.bool)
        else:
            masks[key] = torch.ones(batch_size, dtype=torch.bool)
    
    return masks


def set_visual_default(item: Dict) -> Dict:
    """设置可视化默认值"""
    item = item.copy()
    for elem in item.get('elements', []):
        if 'color' not in elem or elem['color'] is None:
            elem['color'] = [0, 0, 0]
        if 'opacity' not in elem or elem['opacity'] is None:
            elem['opacity'] = 1.0
        if 'font_family' not in elem or elem['font_family'] is None:
            elem['font_family'] = 'DummyFont'
    return item


def tensor_to_list(data: Dict) -> List[Dict]:
    """将批次张量转换为样本列表"""
    batch_size = data['length'].size(0)
    items = []
    
    for i in range(batch_size):
        item = {
            'id': data['id'][i] if 'id' in data else f'sample_{i}',
            'canvas_width': data['canvas_width'][i].item() if 'canvas_width' in data else 800,
            'canvas_height': data['canvas_height'][i].item() if 'canvas_height' in data else 600,
            'length': data['length'][i].item(),
            'elements': []
        }
        
        length = item['length'] + 1
        
        for j in range(length):
            element = {}
            
            for key, value in data.items():
                if key in ['id', 'length', 'canvas_width', 'canvas_height']:
                    continue
                
                if not torch.is_tensor(value):
                    continue
                    
                if value.dim() >= 2 and value.size(1) > j:
                    elem_value = value[i, j]
                    
                    if elem_value.dim() == 0:
                        element[key] = elem_value.item()
                    elif elem_value.dim() == 1:
                        if elem_value.size(0) == 1:
                            element[key] = elem_value[0].item()
                        else:
                            element[key] = elem_value.cpu().numpy().tolist()
                    else:
                        if elem_value.dim() == 2:
                            indices = elem_value.argmax(dim=-1)
                            if indices.size(0) == 1:
                                element[key] = indices[0].item()
                            else:
                                element[key] = indices.cpu().numpy().tolist()
                        else:
                            element[key] = elem_value.cpu().numpy().tolist()
            
            item['elements'].append(element)
        
        items.append(item)
    
    return items


def apply_task_masks(
    example: Dict,
    input_columns: Dict,
    target_task: str,
    attribute_groups: Dict,
    device: str
) -> Dict[str, torch.Tensor]:
    """应用任务特定的掩码"""
    seq_mask = get_seq_mask(example['length'], example['left'].size(1))
    mfp_masks = get_initial_masks(input_columns, seq_mask)
    
    for key in mfp_masks.keys():
        if not input_columns[key].get('is_sequence', False):
            continue
        
        mask = mfp_masks[key].clone()
        
        if target_task == "elem":
            mask[:, 0] = True
        else:
            if key == "type":
                continue
            
            if target_task in attribute_groups:
                attr_keys = attribute_groups[target_task]
                if key in attr_keys:
                    mask = seq_mask.clone()
        
        mfp_masks[key] = mask.to(device)
    
    return mfp_masks


def model_inference_with_masks(model, inputs, masks):
    """使用掩码进行模型推理"""
    masked_inputs = {}
    for key, value in inputs.items():
        if key in masks and torch.is_tensor(value):
            mask = masks[key]
            if mask.any():
                masked_value = value.clone()
                if value.dim() == 3:
                    masked_value[mask] = 0
                masked_inputs[key] = masked_value
            else:
                masked_inputs[key] = value
        else:
            masked_inputs[key] = value
    
    outputs = model(masked_inputs)
    return outputs


def visualize_reconstruction(
    model: torch.nn.Module,
    example: Dict,
    builders: Dict,
    config: DemoConfig,
    input_columns: Dict,
):
    """可视化重建结果"""
    svgs = []
    target_task = config.target_task
    
    items = tensor_to_list(example)
    
    # GT布局和视觉
    svgs.append(list(map(builders["layout"], items)))
    svgs.append(list(map(builders["visual"], items)))
    
    # 输入可视化
    if target_task == "txt":
        svgs.append(list(map(builders["visual_wo_text"], items)))
    elif target_task == "img":
        svgs.append(list(map(builders["visual_wo_image"], items)))
    elif target_task == "attr":
        svgs.append(list(map(builders["visual"], [set_visual_default(x) for x in items])))
    
    # 应用掩码
    mfp_masks = apply_task_masks(
        example, input_columns, target_task,
        config.attribute_groups, config.device
    )
    
    # 元素级任务的特殊处理
    if target_task == "elem":
        example_copy = {}
        for key, value in example.items():
            if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.size(1) > 1:
                indices = torch.where(~mfp_masks[key][0, :])[0]
                example_copy[key] = torch.index_select(value, 1, indices)
            else:
                example_copy[key] = value
        
        example_copy['length'] = example['length'] - 1
        items_copy = tensor_to_list(example_copy)
        svgs.append(list(map(builders["layout"], items_copy)))
        svgs.append(list(map(builders["visual"], items_copy)))
    
    # 模型预测
    with torch.no_grad():
        pred = model_inference_with_masks(model, example, mfp_masks)
    
    for key in example:
        if key not in pred:
            pred[key] = example[key]
    
    pred_items = tensor_to_list(pred)
    
    if target_task in ["pos", "elem"]:
        svgs.append(list(map(builders["layout"], pred_items)))
    svgs.append(list(map(builders["visual"], pred_items)))
    
    # 转置：从按列组织改为按行组织
    return list(zip(*svgs))


# ==================== 主函数 ====================

def main():
    """主函数"""
    config = DemoConfig()
    
    logger.info("="*80)
    logger.info("MFP PyTorch Demo - 图片保存版")
    logger.info("="*80)
    
    # 加载数据
    logger.info(f"Loading dataset from {config.db_root}")
    dataset = DesignLayoutDataset(
        config.db_root,
        split='test',
        max_length=20
    )
    
    from torch.utils.data import DataLoader
    from dataset import collate_fn
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    example = next(iter(dataloader))
    
    for key in example:
        if torch.is_tensor(example[key]):
            example[key] = example[key].to(config.device)
    
    # 加载input_columns配置
    input_columns_file = '/home/dell/Project-HCL/BaseLine/flexdm_pt/scripts/input_columns_generated.json'
    with open(input_columns_file, 'r') as f:
        input_columns = json.load(f)
    
    # 加载模型
    logger.info(f"Loading model from {config.ckpt_dir}")
    checkpoint_path = Path(config.ckpt_dir) / "best_pytorch.pth"
    model = load_model(str(checkpoint_path), input_columns, config.device)
    
    # 构建检索数据库
    logger.info("Building retrieval databases...")
    db_root = Path(config.db_root).parent / config.dataset_name
    
    image_db = ImageRetriever(db_root, image_path=db_root / "images")
    image_db.build("test")
    
    text_db = TextRetriever(db_root, text_path=db_root / "texts")
    text_db.build("test")
    
    # 创建SVG构建器
    logger.info("Creating SVG builders...")
    builders = {}
    
    builders["layout"] = SVGBuilder(
        max_width=128,
        max_height=192,
        key="type",
    )
    
    patterns = [
        ("visual", image_db, text_db),
        ("visual_wo_text", image_db, None),
        ("visual_wo_image", None, text_db),
    ]
    
    for (name, idb, tdb) in patterns:
        builders[name] = SVGBuilder(
            max_width=128,
            max_height=192,
            key="color",
            image_db=idb,
            text_db=tdb,
            render_text=True,
        )
    
    # 可视化重建
    logger.info(f"Visualizing reconstruction for task: {config.target_task}")
    svgs = visualize_reconstruction(
        model, example, builders, config, input_columns
    )
    
    # === 保存结果 ===
    output_base = Path(config.output_dir) / config.target_task
    
    # 1. 分别保存每个样本
    save_visualization_grid(
        svgs=svgs,
        output_dir=output_base / "individual",
        column_names=config.column_names[config.target_task],
        prefix=f"{config.target_task}_sample"
    )
    
    # 2. 创建对比图（只取前8个样本）
    if HAS_CAIROSVG and len(svgs) > 0:
        try:
            create_comparison_image(
                svgs=svgs[:min(8, len(svgs))],
                output_path=output_base / f"comparison_{config.target_task}.png",
                column_names=config.column_names[config.target_task],
                samples_per_row=2,
                svg_width=200,
                svg_height=300,
            )
        except Exception as e:
            logger.error(f"创建对比图失败: {e}")
    
    logger.info("✓ Demo completed!")
    logger.info(f"\n结果保存在: {output_base.absolute()}")


if __name__ == "__main__":
    main()
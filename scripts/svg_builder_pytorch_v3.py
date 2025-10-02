"""
PyTorch版本的SVG构建器（完整修复版）
参考原版 svg_crello.py 的实现，支持正确的坐标系统和元素渲染
"""

import xml.etree.ElementTree as ET
from typing import Dict, Optional, List
import numpy as np
import math
from itertools import chain, groupby

# XML命名空间
NS = {
    "svg": "http://www.w3.org/2000/svg",
    "xlink": "http://www.w3.org/1999/xlink",
}
ET.register_namespace("", NS["svg"])
ET.register_namespace("xlink", NS["xlink"])

# 占位文本
DUMMY_TEXT = "TEXT TEXT TEXT TEXT TEXT TEXT TEXT TEXT TEXT TEXT"


class SVGBuilder:
    """
    SVG构建器 - 用于将布局数据转换为SVG可视化
    
    支持两种模式：
    1. Layout模式 (key='type'): 按元素类型着色，用于查看布局结构
    2. Visual模式 (key='color'): 使用真实颜色，用于查看最终效果
    """
    
    def __init__(
        self,
        key: str = 'type',
        colormap: Optional[Dict] = None,
        preprocessor = None,
        canvas_width: int = 256,
        canvas_height: int = 256,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        opacity: float = 0.5,
        image_db = None,
        text_db = None,
        render_text: bool = False,
        **kwargs
    ):
        """
        Args:
            key: 用于着色的键 ('type' 按类型着色, 'color' 使用真实颜色)
            colormap: 自定义颜色映射字典
            preprocessor: 预处理器（用于自动生成colormap）
            canvas_width: 默认画布宽度
            canvas_height: 默认画布高度
            max_width: 最大显示宽度（用于缩放）
            max_height: 最大显示高度（用于缩放）
            opacity: 不透明度系数
            image_db: 图像检索数据库
            text_db: 文本检索数据库
            render_text: 是否渲染文本内容
        """
        assert key, "key参数不能为空"
        
        self.key = key
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.max_width = max_width
        self.max_height = max_height
        self.opacity = opacity
        self.image_db = image_db
        self.text_db = text_db
        self.render_text = render_text
        
        # 初始化颜色映射
        if key == 'color':
            self.colormap = None  # color模式直接使用元素的color属性
        elif preprocessor is not None:
            vocabulary = preprocessor.get(key, {}).get('vocabulary', [])
            self.colormap = self._make_colormap(vocabulary, colormap)
        elif colormap is not None:
            self.colormap = colormap
        else:
            self.colormap = self._make_default_colormap()
    
    def _make_default_colormap(self) -> Dict:
        """创建默认颜色映射（与原版一致）"""
        return {
            '': 'none',
            'svgElement': 'rgb(66, 166, 246)',      # 蓝色
            'textElement': 'rgb(241, 98, 147)',     # 粉色
            'imageElement': 'rgb(175, 214, 130)',   # 绿色
            'maskElement': 'rgb(79, 196, 248)',     # 青色
            'coloredBackground': 'rgb(226, 191, 232)',  # 紫色
            'videoElement': 'rgb(255, 207, 102)',   # 黄色
            'humanElement': 'rgb(255, 139, 101)',   # 橙色
        }
    
    def _make_colormap(self, vocabulary: List[str], base_colormap=None) -> Dict:
        """根据词汇表自动生成颜色映射"""
        if base_colormap:
            return base_colormap
        
        try:
            from matplotlib import cm
            vocab_size = len(vocabulary)
            cmap = cm.get_cmap('tab20', vocab_size)
            return {
                label: f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})'
                for label, c in zip(vocabulary, cmap(range(vocab_size)))
            }
        except ImportError:
            return self._make_default_colormap()
    
    def compute_canvas_size(self, document: Dict):
        """计算实际画布大小（考虑缩放）"""
        canvas_width = document.get('canvas_width', self.canvas_width)
        canvas_height = document.get('canvas_height', self.canvas_height)
        
        scale = 1.0
        if self.max_width is not None:
            scale = min(self.max_width / canvas_width, scale)
        if self.max_height is not None:
            scale = min(self.max_height / canvas_height, scale)
        
        return canvas_width * scale, canvas_height * scale
    
    def __call__(self, document: Dict) -> str:
        """
        将文档转换为SVG字符串
        
        Args:
            document: 文档字典，必须包含：
                - canvas_width: 画布宽度
                - canvas_height: 画布高度
                - elements: 元素列表，每个元素包含left, top, width, height等属性
        
        Returns:
            SVG格式的字符串
        """
        canvas_width, canvas_height = self.compute_canvas_size(document)
        
        # 创建SVG根元素 - 关键：使用viewBox="0 0 1 1"归一化坐标系
        # 注意：不要在属性中重复声明xmlns，ET.register_namespace已经处理
        root = ET.Element(
            ET.QName(NS["svg"], "svg"),
            {
                'width': str(int(canvas_width)),
                'height': str(int(canvas_height)),
                'viewBox': '0 0 1 1',  # 归一化坐标系：所有坐标在[0,1]范围
                'style': 'background-color: #FFF',
                'preserveAspectRatio': 'none',
            }
        )
        
        # 手动设置命名空间（避免重复属性错误）
        root.set('xmlns', NS["svg"])
        root.set('xmlns:xlink', NS["xlink"])
        
        doc_size = {
            'width': document.get('canvas_width', self.canvas_width),
            'height': document.get('canvas_height', self.canvas_height),
        }
        
        # 添加所有元素
        elements = document.get('elements', [])
        for i, element in enumerate(elements):
            self._add_element(root, element, doc_size, i)
        
        # 转换为字符串
        return ET.tostring(root, encoding='unicode')
    
    def _add_element(self, parent, element: Dict, doc_size: Dict, index: int):
        """添加单个元素到SVG"""
        # 确定填充颜色
        if self.key == 'color':
            # Visual模式：使用元素的真实颜色
            color = element.get('color', [0, 0, 0])
            if isinstance(color, (list, tuple, np.ndarray)):
                fill = f'rgb({int(color[0])},{int(color[1])},{int(color[2])})'
            else:
                fill = 'rgb(0,0,0)'
        else:
            # Layout模式：根据类型从colormap获取颜色
            element_type = element.get(self.key, '')
            
            # 处理各种可能的类型格式
            if isinstance(element_type, (list, tuple)):
                element_type = element_type[0] if len(element_type) > 0 else ''
            if isinstance(element_type, (int, float, np.integer, np.floating)):
                element_type = str(int(element_type))
            if isinstance(element_type, bytes):
                element_type = element_type.decode('utf-8')
            
            fill = self.colormap.get(element_type, 'rgb(128,128,128)')
        
        # 获取位置和尺寸（确保是浮点数，范围在[0,1]）
        left = float(element.get('left', 0))
        top = float(element.get('top', 0))
        width = float(element.get('width', 0.1))
        height = float(element.get('height', 0.1))
        opacity_val = float(element.get('opacity', 1.0))
        
        # 检查是否有图像URL
        image_url = None
        if self.image_db:
            elem_type = element.get('type', '')
            if isinstance(elem_type, bytes):
                elem_type = elem_type.decode('utf-8')
            
            # 检查是否符合图像元素条件
            if elem_type in self.image_db.condition.get('values', []):
                if self.image_db.value in element:
                    image_url = self.image_db.search(element[self.image_db.value])
        
        # 检查是否有文本内容
        text_content = None
        if self.text_db:
            elem_type = element.get('type', '')
            if isinstance(elem_type, bytes):
                elem_type = elem_type.decode('utf-8')
            
            if elem_type in self.text_db.condition.get('values', []):
                if self.text_db.value in element:
                    text_content = self.text_db.search(element[self.text_db.value])
                else:
                    text_content = DUMMY_TEXT
            else:
                text_content = DUMMY_TEXT if elem_type == 'textElement' else None
        elif element.get('type') == 'textElement' and self.render_text:
            text_content = DUMMY_TEXT
        
        # 获取元素ID
        uuid = element.get('uuid', f'elem_{index}')
        if isinstance(uuid, bytes):
            uuid = uuid.decode('utf-8')
        
        # 根据元素类型创建不同的SVG节点
        if image_url and image_url != '':
            node = self._make_image(parent, element, image_url, uuid)
        elif self.render_text and text_content:
            node = self._make_text_element(parent, element, fill, doc_size, text_content, uuid)
        else:
            node = self._make_rect(parent, element, fill, uuid)
        
        # 添加title元素（鼠标悬停显示信息）
        title = ET.SubElement(node, 'title')
        title.text = str({
            k: v for k, v in element.items() 
            if not isinstance(v, (list, np.ndarray)) or len(str(v)) < 50
        })
    
    def _make_rect(self, parent, element: Dict, fill: str, uuid: str):
        """创建矩形元素"""
        return ET.SubElement(
            parent,
            'rect',
            {
                'id': uuid,
                'class': str(element.get('type', '')),
                'x': f'{element.get("left", 0):.6f}',
                'y': f'{element.get("top", 0):.6f}',
                'width': f'{element.get("width", 0.1):.6f}',
                'height': f'{element.get("height", 0.1):.6f}',
                'fill': fill,
                'opacity': f'{element.get("opacity", 1.0) * self.opacity:.3f}',
            }
        )
    
    def _make_image(self, parent, element: Dict, image_url: str, uuid: str):
        """创建图像元素"""
        img = ET.SubElement(
            parent,
            'image',
            {
                'id': uuid,
                'class': str(element.get('type', '')),
                'x': f'{element.get("left", 0):.6f}',
                'y': f'{element.get("top", 0):.6f}',
                'width': f'{element.get("width", 0.1):.6f}',
                'height': f'{element.get("height", 0.1):.6f}',
                'opacity': f'{element.get("opacity", 1.0):.3f}',
                'preserveAspectRatio': 'none',
            }
        )
        # xlink:href 需要特殊处理
        img.set('{http://www.w3.org/1999/xlink}href', image_url)
        return img
    
    def _make_text_element(
        self, 
        parent, 
        element: Dict, 
        fill: str, 
        doc_size: Dict, 
        text_str: str,
        uuid: str
    ):
        """创建文本元素（简化版，适用于PyTorch）"""
        # 添加边距避免裁剪
        margin = element.get('height', 0.1) * 0.1
        
        # 创建文本容器（使用普通字符串避免命名空间问题）
        container = ET.SubElement(
            parent,
            'svg',
            {
                'id': uuid,
                'class': str(element.get('type', '')),
                'x': f'{element.get("left", 0):.6f}',
                'y': f'{(element.get("top", 0) - margin):.6f}',
                'width': f'{element.get("width", 0.1):.6f}',
                'height': f'{element.get("height", 0.1) + margin * 2:.6f}',
                'overflow': 'visible',
            }
        )
        
        # 设置透明度
        opacity_val = element.get('opacity', 1.0)
        if opacity_val < 1:
            container.set('opacity', f'{opacity_val:.3f}')
        
        # 文本属性
        font_size = element.get('height', 0.1) * 0.8
        font_family = element.get('font_family', 'Arial')
        if isinstance(font_family, bytes):
            font_family = font_family.decode('utf-8')
        
        # 创建文本元素
        text_elem = ET.SubElement(
            container,
            'text',
            {
                'x': '50%',
                'y': '50%',
                'text-anchor': 'middle',
                'dominant-baseline': 'central',
                'fill': fill,
                'font-size': f'{font_size:.6f}',
                'font-family': font_family,
            }
        )
        
        # 设置文本内容（限制长度）
        display_text = str(text_str)[:100].strip()
        text_elem.text = display_text
        
        return container


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("SVG Builder 测试")
    print("="*60)
    
    # 创建测试文档
    test_doc = {
        'id': 'test_001',
        'canvas_width': 800,
        'canvas_height': 600,
        'elements': [
            {
                'type': 'coloredBackground',
                'left': 0.0,
                'top': 0.0,
                'width': 1.0,
                'height': 1.0,
                'color': [240, 240, 240],
                'opacity': 1.0,
            },
            {
                'type': 'imageElement',
                'left': 0.1,
                'top': 0.1,
                'width': 0.3,
                'height': 0.4,
                'color': [255, 100, 100],
                'opacity': 1.0,
            },
            {
                'type': 'textElement',
                'left': 0.5,
                'top': 0.2,
                'width': 0.4,
                'height': 0.1,
                'color': [100, 100, 255],
                'opacity': 1.0,
                'font_family': 'Arial',
            },
            {
                'type': 'svgElement',
                'left': 0.2,
                'top': 0.6,
                'width': 0.6,
                'height': 0.3,
                'color': [100, 255, 100],
                'opacity': 0.8,
            },
        ]
    }
    
    print("\n测试1: Layout视图（按type着色）")
    builder_layout = SVGBuilder(key='type', max_width=400, opacity=0.8)
    svg_layout = builder_layout(test_doc)
    with open('test_layout.svg', 'w', encoding='utf-8') as f:
        f.write(svg_layout)
    print(f"✓ 生成 Layout SVG: {len(svg_layout)} 字符")
    print(f"  - 保存到: test_layout.svg")
    
    print("\n测试2: Visual视图（使用真实颜色）")
    builder_visual = SVGBuilder(key='color', max_width=400, opacity=1.0, render_text=True)
    svg_visual = builder_visual(test_doc)
    with open('test_visual.svg', 'w', encoding='utf-8') as f:
        f.write(svg_visual)
    print(f"✓ 生成 Visual SVG: {len(svg_visual)} 字符")
    print(f"  - 保存到: test_visual.svg")
    
    print("\n" + "="*60)
    print("✓ 测试完成！请在浏览器中打开SVG文件查看效果")
    print("="*60)
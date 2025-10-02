"""
PyTorch版本的SVG构建器
用于将布局数据转换为SVG可视化
"""

import xml.etree.ElementTree as ET
from typing import Dict, Optional
import numpy as np


class SVGBuilder:
    """SVG构建器"""
    
    def __init__(
        self,
        key: str = 'type',
        colormap: Optional[Dict] = None,
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
            key: 用于着色的键 ('type' 或 'color')
            colormap: 颜色映射字典
            canvas_width: 画布宽度
            canvas_height: 画布高度
            max_width: 最大显示宽度
            max_height: 最大显示高度
            opacity: 不透明度
            image_db: 图像检索数据库
            text_db: 文本检索数据库
            render_text: 是否渲染文本
        """
        self.key = key
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.max_width = max_width
        self.max_height = max_height
        self.opacity = opacity
        self.image_db = image_db
        self.text_db = text_db
        self.render_text = render_text
        
        # 颜色映射
        if colormap is None and key != 'color':
            self.colormap = self._make_default_colormap()
        else:
            self.colormap = colormap or {}
    
    def _make_default_colormap(self) -> Dict:
        """创建默认颜色映射"""
        # 常见元素类型的颜色
        return {
            'imageElement': 'rgb(66, 166, 246)',
            'textElement': 'rgb(241, 98, 147)',
            'svgElement': 'rgb(175, 214, 130)',
            'maskElement': 'rgb(79, 196, 248)',
            'coloredBackground': 'rgb(226, 191, 232)',
            'humanElement': 'rgb(255, 139, 101)',
            '': 'none',
        }
    
    def compute_canvas_size(self, document: Dict):
        """计算画布大小"""
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
            document: 文档字典
        
        Returns:
            SVG字符串
        """
        canvas_width, canvas_height = self.compute_canvas_size(document)
        
        # 创建SVG根元素
        root = ET.Element(
            'svg',
            {
                'width': str(int(canvas_width)),
                'height': str(int(canvas_height)),
                'viewBox': '0 0 1 1',
                'style': 'background-color: #FFF',
                'preserveAspectRatio': 'none',
                'xmlns': 'http://www.w3.org/2000/svg',
            }
        )
        
        # 添加元素
        for element in document.get('elements', []):
            self._add_element(root, element, document)
        
        # 转换为字符串
        return ET.tostring(root, encoding='unicode')
    
    def _add_element(self, parent, element: Dict, document: Dict):
        """添加单个元素到SVG"""
        # 获取颜色
        if self.key == 'color':
            color = element.get('color', [0, 0, 0])
            if isinstance(color, list):
                fill = f'rgb({int(color[0])},{int(color[1])},{int(color[2])})'
            else:
                fill = 'rgb(0,0,0)'
        else:
            element_type = element.get(self.key, '')
            fill = self.colormap.get(element_type, 'rgb(128,128,128)')
        
        # 获取位置和尺寸
        left = float(element.get('left', 0))
        top = float(element.get('top', 0))
        width = float(element.get('width', 0.1))
        height = float(element.get('height', 0.1))
        opacity = float(element.get('opacity', 1.0))
        
        # 检查是否需要渲染图像或文本
        image_url = None
        if self.image_db and element.get('type') in ['imageElement', 'svgElement', 'maskElement']:
            if 'image_embedding' in element:
                image_url = self.image_db.search(element['image_embedding'])
        
        text_content = None
        if self.text_db and element.get('type') == 'textElement':
            if 'text_embedding' in element:
                text_content = self.text_db.search(element['text_embedding'])
        
        # 创建元素
        if image_url and image_url != '':
            # 图像元素
            elem = ET.SubElement(
                parent,
                'image',
                {
                    'x': str(left),
                    'y': str(top),
                    'width': str(width),
                    'height': str(height),
                    'href': image_url,
                    'opacity': str(opacity * self.opacity),
                    'preserveAspectRatio': 'none',
                }
            )
        elif self.render_text and text_content:
            # 文本元素
            container = ET.SubElement(
                parent,
                'svg',
                {
                    'x': str(left),
                    'y': str(top),
                    'width': str(width),
                    'height': str(height),
                    'overflow': 'visible',
                }
            )
            
            text_elem = ET.SubElement(
                container,
                'text',
                {
                    'x': '50%',
                    'y': '50%',
                    'text-anchor': 'middle',
                    'dominant-baseline': 'middle',
                    'fill': fill,
                    'font-size': str(height * 0.8),
                    'font-family': element.get('font_family', 'Arial'),
                }
            )
            text_elem.text = str(text_content)[:50]  # 限制长度
        else:
            # 矩形元素
            elem = ET.SubElement(
                parent,
                'rect',
                {
                    'x': str(left),
                    'y': str(top),
                    'width': str(width),
                    'height': str(height),
                    'fill': fill,
                    'opacity': str(opacity * self.opacity),
                }
            )
        
        # 添加标题（用于hover显示）
        title = ET.SubElement(elem if not (self.render_text and text_content) else container, 'title')
        title.text = str({k: v for k, v in element.items() if not isinstance(v, (list, np.ndarray))})


# 测试代码
if __name__ == "__main__":
    # 创建测试文档
    test_doc = {
        'id': 'test_001',
        'canvas_width': 800,
        'canvas_height': 600,
        'length': 3,
        'elements': [
            {
                'type': 'imageElement',
                'left': 0.1,
                'top': 0.1,
                'width': 0.3,
                'height': 0.3,
                'color': [255, 0, 0],
                'opacity': 1.0,
            },
            {
                'type': 'textElement',
                'left': 0.5,
                'top': 0.5,
                'width': 0.3,
                'height': 0.1,
                'color': [0, 0, 255],
                'opacity': 1.0,
                'font_family': 'Arial',
            },
            {
                'type': 'coloredBackground',
                'left': 0.0,
                'top': 0.0,
                'width': 1.0,
                'height': 1.0,
                'color': [240, 240, 240],
                'opacity': 0.5,
            },
        ]
    }
    
    # 测试布局构建器
    builder = SVGBuilder(key='type', max_width=400)
    svg = builder(test_doc)
    print("SVG生成成功!")
    print(f"SVG长度: {len(svg)} 字符")
    
    # 保存到文件
    with open('test_layout.svg', 'w') as f:
        f.write(svg)
    print("✓ 已保存到 test_layout.svg")
"""
PyTorch版本的SVG构建器（简化版 - 避免命名空间问题）
"""

from typing import Dict, Optional, List
import numpy as np


class SVGBuilder:
    """SVG构建器 - 简化版，使用字符串模板避免XML命名空间问题"""
    
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
            self.colormap = None
        elif preprocessor is not None:
            vocabulary = preprocessor.get(key, {}).get('vocabulary', [])
            self.colormap = self._make_colormap(vocabulary, colormap)
        elif colormap is not None:
            self.colormap = colormap
        else:
            self.colormap = self._make_default_colormap()
    
    def _make_default_colormap(self) -> Dict:
        """创建默认颜色映射"""
        return {
            '': 'none',
            'svgElement': 'rgb(66, 166, 246)',
            'textElement': 'rgb(241, 98, 147)',
            'imageElement': 'rgb(175, 214, 130)',
            'maskElement': 'rgb(79, 196, 248)',
            'coloredBackground': 'rgb(226, 191, 232)',
            'videoElement': 'rgb(255, 207, 102)',
            'humanElement': 'rgb(255, 139, 101)',
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
        """计算实际画布大小"""
        canvas_width = document.get('canvas_width', self.canvas_width)
        canvas_height = document.get('canvas_height', self.canvas_height)
        
        scale = 1.0
        if self.max_width is not None:
            scale = min(self.max_width / canvas_width, scale)
        if self.max_height is not None:
            scale = min(self.max_height / canvas_height, scale)
        
        return canvas_width * scale, canvas_height * scale
    
    def __call__(self, document: Dict) -> str:
        """将文档转换为SVG字符串（使用字符串模板）"""
        canvas_width, canvas_height = self.compute_canvas_size(document)
        
        # SVG头部
        svg_parts = [
            f'<svg width="{int(canvas_width)}" height="{int(canvas_height)}" '
            f'viewBox="0 0 1 1" style="background-color: #FFF" '
            f'preserveAspectRatio="none" '
            f'xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink">'
        ]
        
        # 添加元素
        elements = document.get('elements', [])
        for i, element in enumerate(elements):
            svg_parts.append(self._element_to_svg(element, i))
        
        # SVG尾部
        svg_parts.append('</svg>')
        
        return '\n'.join(svg_parts)
    
    def _element_to_svg(self, element: Dict, index: int) -> str:
        """将单个元素转换为SVG字符串"""
        # 获取颜色
        if self.key == 'color':
            color = element.get('color', [0, 0, 0])
            if isinstance(color, (list, tuple, np.ndarray)):
                fill = f'rgb({int(color[0])},{int(color[1])},{int(color[2])})'
            else:
                fill = 'rgb(0,0,0)'
        else:
            element_type = element.get(self.key, '')
            if isinstance(element_type, (list, tuple)):
                element_type = element_type[0] if len(element_type) > 0 else ''
            if isinstance(element_type, (int, float, np.integer, np.floating)):
                element_type = str(int(element_type))
            if isinstance(element_type, bytes):
                element_type = element_type.decode('utf-8')
            fill = self.colormap.get(element_type, 'rgb(128,128,128)')
        
        # 获取位置和尺寸
        left = float(element.get('left', 0))
        top = float(element.get('top', 0))
        width = float(element.get('width', 0.1))
        height = float(element.get('height', 0.1))
        opacity_val = float(element.get('opacity', 1.0))
        
        # 元素ID和类型
        uuid = element.get('uuid', f'elem_{index}')
        if isinstance(uuid, bytes):
            uuid = uuid.decode('utf-8')
        elem_type = str(element.get('type', ''))
        
        # 检查图像
        image_url = None
        if self.image_db:
            et = element.get('type', '')
            if isinstance(et, bytes):
                et = et.decode('utf-8')
            if et in self.image_db.condition.get('values', []):
                if self.image_db.value in element:
                    image_url = self.image_db.search(element[self.image_db.value])
        
        # 检查文本
        text_content = None
        if self.text_db or self.render_text:
            et = element.get('type', '')
            if isinstance(et, bytes):
                et = et.decode('utf-8')
            if et == 'textElement':
                if self.text_db and self.text_db.value in element:
                    text_content = self.text_db.search(element[self.text_db.value])
                else:
                    text_content = "TEXT TEXT TEXT"
        
        # 生成SVG
        if image_url and image_url != '':
            return self._make_image_svg(uuid, elem_type, left, top, width, height, opacity_val, image_url)
        elif self.render_text and text_content:
            return self._make_text_svg(uuid, elem_type, left, top, width, height, opacity_val, fill, text_content, element)
        else:
            return self._make_rect_svg(uuid, elem_type, left, top, width, height, opacity_val * self.opacity, fill)
    
    def _make_rect_svg(self, uuid: str, elem_type: str, left: float, top: float, 
                       width: float, height: float, opacity: float, fill: str) -> str:
        """创建矩形SVG"""
        return (
            f'<rect id="{uuid}" class="{elem_type}" '
            f'x="{left:.6f}" y="{top:.6f}" '
            f'width="{width:.6f}" height="{height:.6f}" '
            f'fill="{fill}" opacity="{opacity:.3f}"/>'
        )
    
    def _make_image_svg(self, uuid: str, elem_type: str, left: float, top: float,
                        width: float, height: float, opacity: float, url: str) -> str:
        """创建图像SVG"""
        return (
            f'<image id="{uuid}" class="{elem_type}" '
            f'x="{left:.6f}" y="{top:.6f}" '
            f'width="{width:.6f}" height="{height:.6f}" '
            f'xlink:href="{url}" opacity="{opacity:.3f}" '
            f'preserveAspectRatio="none"/>'
        )
    
    def _make_text_svg(self, uuid: str, elem_type: str, left: float, top: float,
                       width: float, height: float, opacity: float, fill: str,
                       text: str, element: Dict) -> str:
        """创建文本SVG"""
        margin = height * 0.1
        font_size = height * 0.8
        font_family = element.get('font_family', 'Arial')
        if isinstance(font_family, bytes):
            font_family = font_family.decode('utf-8')
        
        display_text = str(text)[:100].strip()
        # 转义XML特殊字符
        display_text = display_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        opacity_str = f' opacity="{opacity:.3f}"' if opacity < 1 else ''
        
        return (
            f'<svg id="{uuid}" class="{elem_type}" '
            f'x="{left:.6f}" y="{(top - margin):.6f}" '
            f'width="{width:.6f}" height="{(height + margin * 2):.6f}" '
            f'overflow="visible"{opacity_str}>'
            f'<text x="50%" y="50%" text-anchor="middle" dominant-baseline="central" '
            f'fill="{fill}" font-size="{font_size:.6f}" font-family="{font_family}">'
            f'{display_text}</text></svg>'
        )


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("SVG Builder 测试（简化版）")
    print("="*60)
    
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
        ]
    }
    
    print("\n测试1: Layout视图")
    builder_layout = SVGBuilder(key='type', max_width=400, opacity=0.8)
    svg_layout = builder_layout(test_doc)
    with open('test_layout.svg', 'w', encoding='utf-8') as f:
        f.write(svg_layout)
    print(f"✓ 生成 {len(svg_layout)} 字符")
    
    print("\n测试2: Visual视图")
    builder_visual = SVGBuilder(key='color', max_width=400, opacity=1.0, render_text=True)
    svg_visual = builder_visual(test_doc)
    with open('test_visual.svg', 'w', encoding='utf-8') as f:
        f.write(svg_visual)
    print(f"✓ 生成 {len(svg_visual)} 字符")
    
    print("\n" + "="*60)
    print("✓ 测试完成！")
    print("="*60)
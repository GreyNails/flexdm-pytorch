"""
PyTorch版本的检索器
用于图像和文本的最近邻检索
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from base64 import b64encode

import torch
import numpy as np

# 使用faiss进行快速检索
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("Faiss not available, using brute force search")

logger = logging.getLogger(__name__)


class BaseRetriever:
    """基础检索器"""
    
    def __init__(
        self,
        data_path: Path,
        key: str,
        value: str,
        condition: Dict[str, Any] = None,
        dim: int = 512,
    ):
        """
        Args:
            data_path: 数据路径
            key: 查询键
            value: 检索值
            condition: 条件过滤
            dim: 嵌入维度
        """
        self.data_path = Path(data_path)
        self.key = key
        self.value = value
        self.condition = condition
        self.dim = dim
        
        self.labels = None
        self.db = None
    
    def build(self, split: str = 'train'):
        """
        构建检索索引
        
        Args:
            split: 数据集划分
        """
        logger.info(f"Building {self.__class__.__name__} index for {split}...")
        
        # 加载数据
        json_file = self.data_path / f"{split}.json"
        if not json_file.exists():
            logger.warning(f"Data file not found: {json_file}")
            return
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取嵌入和标签
        embeddings = []
        labels = []
        
        for item in data:
            length = item['length']
            
            for i in range(length):
                # 检查条件
                if self.condition:
                    cond_key = self.condition['key']
                    cond_values = self.condition['values']
                    if item[cond_key][i] not in cond_values:
                        continue
                
                # 提取嵌入和标签
                if self.key in item and self.value in item:
                    key_val = item[self.key][i]
                    value_val = item[self.value][i]
                    
                    if isinstance(value_val, list) and len(value_val) == self.dim:
                        embeddings.append(value_val)
                        labels.append(key_val)
        
        if not embeddings:
            logger.warning(f"No embeddings found for {split}")
            return
        
        # 去重
        unique_data = {}
        for label, emb in zip(labels, embeddings):
            if label not in unique_data:
                unique_data[label] = emb
        
        self.labels = np.array(list(unique_data.keys()))
        embeddings = np.array(list(unique_data.values()), dtype=np.float32)
        
        # 构建索引
        if FAISS_AVAILABLE:
            self.db = faiss.IndexFlatL2(self.dim)
            self.db.add(embeddings)
        else:
            # 使用PyTorch进行暴力搜索
            self.db = torch.from_numpy(embeddings)
        
        logger.info(f"✓ Built index with {len(self.labels)} items")
    
    def search(self, query, k: int = 1):
        """
        搜索最近邻
        
        Args:
            query: 查询向量
            k: 返回的最近邻数量
        
        Returns:
            检索结果
        """
        if self.labels is None or self.db is None:
            return self.get_default_result()
        
        # 转换查询为numpy
        if torch.is_tensor(query):
            query = query.cpu().numpy()
        if not isinstance(query, np.ndarray):
            query = np.array(query, dtype=np.float32)
        
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # 搜索
        if FAISS_AVAILABLE:
            _, indices = self.db.search(query, k)
        else:
            # PyTorch暴力搜索
            query_torch = torch.from_numpy(query)
            distances = torch.cdist(query_torch, self.db)
            _, indices = distances.topk(k, largest=False)
            indices = indices.cpu().numpy()
        
        # 获取结果
        results = [self.get_url(idx) for idx in indices[0]]
        
        return results[0] if k == 1 else results
    
    def get_url(self, index: int) -> str:
        """获取URL（子类实现）"""
        raise NotImplementedError
    
    def get_default_result(self):
        """获取默认结果"""
        return ""


class ImageRetriever(BaseRetriever):
    """图像检索器"""
    
    def __init__(
        self,
        data_path: Path,
        key: str = 'image_hash',
        value: str = 'image_embedding',
        condition: Dict[str, Any] = None,
        image_path: Path = None,
        dim: int = 512,
    ):
        """
        Args:
            data_path: 数据路径
            key: 图像哈希键
            value: 图像嵌入键
            condition: 条件过滤
            image_path: 图像文件路径
            dim: 嵌入维度
        """
        super().__init__(data_path, key, value, condition, dim)
        
        if condition is None:
            self.condition = {
                'key': 'type',
                'values': ['imageElement', 'maskElement', 'svgElement', 'humanElement'],
            }
        
        self.image_path = image_path or self.data_path / 'images'
    
    def get_url(self, index: int) -> str:
        """获取图像URL"""
        label = self.labels[index]
        
        if isinstance(label, bytes):
            label = label.decode('utf-8')
        
        if label:
            image_file = self.image_path / f"{label}.png"
            if image_file.exists():
                return self._make_data_uri(image_file)
        
        return ""
    
    def _make_data_uri(self, file_path: Path, mime_type: str = 'image/png') -> str:
        """创建data URI"""
        try:
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
            data = b64encode(image_bytes).decode('ascii')
            return f"data:{mime_type};base64,{data}"
        except Exception as e:
            logger.warning(f"Failed to read image {file_path}: {e}")
            return ""


class TextRetriever(BaseRetriever):
    """文本检索器"""
    
    def __init__(
        self,
        data_path: Path,
        key: str = 'text_hash',
        value: str = 'text_embedding',
        condition: Dict[str, Any] = None,
        text_path: Path = None,
        dim: int = 512,
    ):
        """
        Args:
            data_path: 数据路径
            key: 文本哈希键
            value: 文本嵌入键
            condition: 条件过滤
            text_path: 文本文件路径
            dim: 嵌入维度
        """
        super().__init__(data_path, key, value, condition, dim)
        
        if condition is None:
            self.condition = {
                'key': 'type',
                'values': ['textElement'],
            }
        
        self.text_path = text_path or self.data_path / 'texts'
    
    def get_url(self, index: int) -> str:
        """获取文本内容"""
        label = self.labels[index]
        
        if isinstance(label, bytes):
            label = label.decode('utf-8')
        
        if label:
            text_file = self.text_path / f"{label}.txt"
            if text_file.exists():
                try:
                    with open(text_file, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    logger.warning(f"Failed to read text {text_file}: {e}")
        
        return "TEXT"


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试路径
    data_path = Path("./data/crello_json")
    
    if data_path.exists():
        # 测试图像检索
        print("测试图像检索器...")
        image_retriever = ImageRetriever(
            data_path,
            image_path=data_path.parent / "crello" / "images"
        )
        image_retriever.build("test")
        
        # 测试查询
        if image_retriever.labels is not None:
            test_query = np.random.randn(512).astype(np.float32)
            result = image_retriever.search(test_query)
            print(f"图像检索结果长度: {len(result)}")
        
        # 测试文本检索
        print("\n测试文本检索器...")
        text_retriever = TextRetriever(
            data_path,
            text_path=data_path.parent / "crello" / "texts"
        )
        text_retriever.build("test")
        
        # 测试查询
        if text_retriever.labels is not None:
            test_query = np.random.randn(512).astype(np.float32)
            result = text_retriever.search(test_query)
            print(f"文本检索结果: {result[:50]}...")
        
        print("\n✓ 测试完成!")
    else:
        print(f"数据路径不存在: {data_path}")
"""
TFRecord 转 JSON 转换脚本
用于将Crello数据集从TFRecord格式转换为JSON格式
"""

import tensorflow as tf
import json
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np


class TFRecordToJSONConverter:
    def __init__(self, tfrecord_dir, output_dir, dataset_name='crello'):
        self.tfrecord_dir = Path(tfrecord_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        
        # 定义特征描述（根据crello-spec.yml）
        self.context_features = {
            'id': tf.io.FixedLenFeature([], tf.string),
            'group': tf.io.FixedLenFeature([], tf.string),
            'format': tf.io.FixedLenFeature([], tf.string),
            'canvas_width': tf.io.FixedLenFeature([], tf.int64),
            'canvas_height': tf.io.FixedLenFeature([], tf.int64),
            'category': tf.io.FixedLenFeature([], tf.string),
            'length': tf.io.FixedLenFeature([], tf.int64),
        }
        
        self.sequence_features = {
            'uuid': tf.io.FixedLenSequenceFeature([], tf.string),
            'type': tf.io.FixedLenSequenceFeature([], tf.string),
            'left': tf.io.FixedLenSequenceFeature([], tf.float32),
            'top': tf.io.FixedLenSequenceFeature([], tf.float32),
            'width': tf.io.FixedLenSequenceFeature([], tf.float32),
            'height': tf.io.FixedLenSequenceFeature([], tf.float32),
            'opacity': tf.io.FixedLenSequenceFeature([], tf.float32),
            'color': tf.io.FixedLenSequenceFeature([3], tf.int64),
            'font_family': tf.io.FixedLenSequenceFeature([], tf.string),
            'image_embedding': tf.io.FixedLenSequenceFeature([512], tf.float32),
            'text_embedding': tf.io.FixedLenSequenceFeature([512], tf.float32),
        }

    def parse_tfrecord_example(self, serialized_example):
        """解析单个TFRecord样本"""
        context, sequence, _ = tf.io.parse_sequence_example(
            serialized_example,
            context_features=self.context_features,
            sequence_features=self.sequence_features
        )
        return context, sequence

    def tensor_to_python(self, tensor):
        """将TensorFlow张量转换为Python原生类型"""
        if isinstance(tensor, tf.Tensor):
            tensor = tensor.numpy()
        
        if isinstance(tensor, bytes):
            return tensor.decode('utf-8')
        elif isinstance(tensor, np.ndarray):
            if tensor.dtype == np.object_:  # string array
                return [x.decode('utf-8') if isinstance(x, bytes) else x for x in tensor]
            else:
                return tensor.tolist()
        elif isinstance(tensor, (np.int64, np.int32)):
            return int(tensor)
        elif isinstance(tensor, (np.float32, np.float64)):
            return float(tensor)
        else:
            return tensor

    def convert_example_to_dict(self, context, sequence):
        """将解析的样本转换为字典"""
        data = {}
        
        # 文档级别特征
        for key, value in context.items():
            data[key] = self.tensor_to_python(value)
        
        # 元素级别特征
        length = int(context['length'].numpy())
        for key, value in sequence.items():
            data[key] = self.tensor_to_python(value)
            # 只保留有效长度的数据
            if isinstance(data[key], list) and len(data[key]) > length:
                data[key] = data[key][:length]
        
        return data

    def convert_split(self, split='train'):
        """转换指定split的数据"""
        print(f"\n转换 {split} split...")
        
        # 查找所有tfrecord文件
        pattern = f"{split}-*.tfrecord"
        tfrecord_files = sorted(self.tfrecord_dir.glob(pattern))
        
        if not tfrecord_files:
            print(f"警告: 未找到匹配 {pattern} 的文件")
            return
        
        all_data = []
        
        for tfrecord_file in tfrecord_files:
            print(f"处理文件: {tfrecord_file.name}")
            
            # 创建数据集
            dataset = tf.data.TFRecordDataset(str(tfrecord_file))
            
            # 处理每个样本
            for serialized in tqdm(dataset):
                try:
                    context, sequence = self.parse_tfrecord_example(serialized)
                    data_dict = self.convert_example_to_dict(context, sequence)
                    all_data.append(data_dict)
                except Exception as e:
                    print(f"解析错误: {e}")
                    continue
        
        # 保存为JSON
        output_file = self.output_dir / f"{split}.json"
        print(f"保存到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 完成! 共 {len(all_data)} 个样本")
        return len(all_data)

    def convert_all_splits(self):
        """转换所有split"""
        splits = ['train', 'val', 'test']
        stats = {}
        
        for split in splits:
            count = self.convert_split(split)
            if count:
                stats[split] = count
        
        # 保存统计信息
        with open(self.output_dir / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("\n转换完成!")
        print(f"统计信息: {stats}")


def main():
    # 配置路径
    TFRECORD_DIR = "/home/dell/Project-HCL/BaseLine/flexdm_pt/data/crello"
    OUTPUT_DIR = "/home/dell/Project-HCL/BaseLine/flexdm_pt/data/crello_json"
    
    # 创建转换器
    converter = TFRecordToJSONConverter(TFRECORD_DIR, OUTPUT_DIR)
    
    # 转换所有数据
    converter.convert_all_splits()


if __name__ == "__main__":
    main()
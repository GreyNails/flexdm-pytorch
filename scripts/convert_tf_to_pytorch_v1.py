"""
TensorFlow Checkpoint 转 PyTorch 权重转换脚本
将训练好的TF模型权重转换为PyTorch格式
"""

import tensorflow as tf
import torch
import numpy as np
from pathlib import Path
import re
from collections import OrderedDict


class TFToPyTorchConverter:
    def __init__(self, tf_checkpoint_path, output_path):
        self.tf_checkpoint_path = tf_checkpoint_path
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def load_tf_checkpoint(self):
        """加载TensorFlow checkpoint"""
        print(f"加载TF checkpoint: {self.tf_checkpoint_path}")
        reader = tf.train.load_checkpoint(self.tf_checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        
        weights = {}
        for key in sorted(var_to_shape_map.keys()):
            try:
                weights[key] = reader.get_tensor(key)
                print(f"✓ 加载: {key}, shape: {weights[key].shape}")
            except Exception as e:
                print(f"✗ 跳过: {key}, 错误: {e}")
        
        return weights
    
    def convert_layer_name(self, tf_name):
        """
        将TensorFlow层名转换为PyTorch命名约定
        
        TF: model/encoder/input_left/embeddings:0
        PT: model.encoder.input_left.weight
        """
        # 移除:0后缀
        name = tf_name.split(':')[0]
        
        # 替换/为.
        name = name.replace('/', '.')
        
        # 转换常见的TF层名到PyTorch
        conversions = {
            'kernel': 'weight',
            'bias': 'bias',
            'gamma': 'weight',  # LayerNorm
            'beta': 'bias',     # LayerNorm
            'embeddings': 'weight',
        }
        
        for tf_term, pt_term in conversions.items():
            name = name.replace(tf_term, pt_term)
        
        return name
    
    def transpose_if_needed(self, name, tensor):
        """
        根据层类型决定是否需要转置
        TensorFlow: [in_features, out_features]
        PyTorch: [out_features, in_features]
        """
        # Dense/Linear层需要转置
        if 'dense' in name.lower() or 'fc' in name.lower():
            if len(tensor.shape) == 2 and 'weight' in name:
                return tensor.T
        
        # Embedding层不需要转置
        # LayerNorm不需要转置
        
        return tensor
    
    def convert_attention_weights(self, weights_dict):
        """
        转换注意力层权重
        TF的MultiHeadAttention与PyTorch的nn.MultiheadAttention格式不同
        """
        converted = {}
        
        # 查找所有attention相关的权重
        attn_keys = [k for k in weights_dict.keys() if 'attn' in k.lower()]
        
        for key in attn_keys:
            pt_key = self.convert_layer_name(key)
            tensor = weights_dict[key]
            tensor = self.transpose_if_needed(pt_key, tensor)
            converted[pt_key] = torch.from_numpy(tensor)
        
        return converted
    
    def convert_to_pytorch(self):
        """执行完整转换"""
        # 加载TF权重
        tf_weights = self.load_tf_checkpoint()
        
        # 转换为PyTorch格式
        pytorch_state_dict = OrderedDict()
        
        print("\n开始转换权重...")
        for tf_name, tf_tensor in tf_weights.items():
            # 转换名称
            pt_name = self.convert_layer_name(tf_name)
            
            # 转换数据
            pt_tensor = self.transpose_if_needed(pt_name, tf_tensor)
            pt_tensor = torch.from_numpy(pt_tensor.astype(np.float32))
            
            pytorch_state_dict[pt_name] = pt_tensor
            print(f"  {tf_name} -> {pt_name}, shape: {pt_tensor.shape}")
        
        # 保存PyTorch权重
        print(f"\n保存PyTorch权重到: {self.output_path}")
        torch.save({
            'state_dict': pytorch_state_dict,
            'conversion_info': {
                'source': str(self.tf_checkpoint_path),
                'num_params': len(pytorch_state_dict),
            }
        }, self.output_path)
        
        print(f"✓ 转换完成! 共 {len(pytorch_state_dict)} 个参数")
        
        # 输出摘要
        self.print_summary(pytorch_state_dict)
        
        return pytorch_state_dict
    
    def print_summary(self, state_dict):
        """打印权重摘要"""
        print("\n" + "="*60)
        print("权重摘要")
        print("="*60)
        
        total_params = 0
        layer_groups = {}
        
        for name, tensor in state_dict.items():
            # 统计参数
            params = tensor.numel()
            total_params += params
            
            # 按模块分组
            module = name.split('.')[0] if '.' in name else 'root'
            if module not in layer_groups:
                layer_groups[module] = {'count': 0, 'params': 0}
            layer_groups[module]['count'] += 1
            layer_groups[module]['params'] += params
        
        print(f"\n总参数数: {total_params:,}")
        print(f"总层数: {len(state_dict)}")
        print("\n按模块统计:")
        for module, stats in sorted(layer_groups.items()):
            print(f"  {module:20s}: {stats['count']:3d} 层, "
                  f"{stats['params']:10,} 参数")


def convert_checkpoint(checkpoint_dir, checkpoint_name='best.ckpt'):
    """
    便捷函数：转换checkpoint
    
    Args:
        checkpoint_dir: checkpoint目录
        checkpoint_name: checkpoint文件名
    """
    checkpoint_path = Path(checkpoint_dir) / checkpoint_name
    output_path = Path(checkpoint_dir) / f"{checkpoint_name.replace('.ckpt', '')}_pytorch.pth"
    
    converter = TFToPyTorchConverter(str(checkpoint_path), str(output_path))
    return converter.convert_to_pytorch()


def main():
    # 配置路径
    CHECKPOINT_DIR = "/home/dell/Project-HCL/BaseLine/flex-dm/tmp/hc/checkpoints"
    
    # 转换best checkpoint
    print("转换best checkpoint...")
    convert_checkpoint(CHECKPOINT_DIR, 'best.ckpt')
    
    print("\n" + "="*60)
    
    # 转换final checkpoint
    print("转换final checkpoint...")
    convert_checkpoint(CHECKPOINT_DIR, 'final.ckpt')


if __name__ == "__main__":
    main()
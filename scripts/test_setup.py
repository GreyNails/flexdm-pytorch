"""
测试训练环境设置
运行此脚本以验证一切配置正确
"""

import os
import sys
import json
from pathlib import Path

def check_dependencies():
    """检查依赖包"""
    print("="*60)
    print("检查依赖包...")
    print("="*60)
    
    packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'tqdm': 'tqdm',
        'tensorboard': 'TensorBoard',
    }
    
    missing = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name:20s} 已安装")
        except ImportError:
            print(f"✗ {name:20s} 未安装")
            missing.append(package)
    
    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        print(f"请运行: pip install {' '.join(missing)}")
        return False
    
    # 检查CUDA
    import torch
    if torch.cuda.is_available():
        print(f"\n✓ CUDA 可用")
        print(f"  设备数: {torch.cuda.device_count()}")
        print(f"  当前设备: {torch.cuda.current_device()}")
        print(f"  设备名称: {torch.cuda.get_device_name(0)}")
    else:
        print(f"\n⚠ CUDA 不可用，将使用CPU训练")
    
    return True


def check_data(data_dir):
    """检查数据文件"""
    print("\n" + "="*60)
    print("检查数据文件...")
    print("="*60)
    
    data_dir = Path(data_dir)
    
    required_files = {
        'train.json': '训练数据',
        'val.json': '验证数据',
        'test.json': '测试数据',
        'vocabulary.json': '词汇表',
    }
    
    all_exist = True
    for filename, desc in required_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size / (1024**2)  # MB
            print(f"✓ {desc:15s} 存在 ({size:.2f} MB)")
            
            # 检查JSON格式
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    print(f"  └─ 包含 {len(data)} 个样本")
            except Exception as e:
                print(f"  └─ 警告: 无法读取 ({e})")
        else:
            print(f"✗ {desc:15s} 不存在")
            all_exist = False
    
    return all_exist


def test_data_loading(data_dir):
    """测试数据加载"""
    print("\n" + "="*60)
    print("测试数据加载...")
    print("="*60)
    
    try:
        from dataset import DesignLayoutDataset, create_dataloader
        
        # 创建数据集
        dataset = DesignLayoutDataset(data_dir, split='train', max_length=20)
        print(f"✓ 数据集创建成功")
        print(f"  样本数: {len(dataset)}")
        
        # 测试单个样本
        sample = dataset[0]
        print(f"\n样本结构:")
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"  {key:20s}: shape={list(value.shape)}, dtype={value.dtype}")
            else:
                print(f"  {key:20s}: {type(value)}")
        
        # 测试dataloader
        loader = create_dataloader(data_dir, 'train', batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        print(f"\n✓ DataLoader测试成功")
        print(f"  批次大小: {batch['left'].shape[0]}")
        
        # 生成input_columns
        input_columns = dataset.get_input_columns()
        print(f"\n✓ Input columns生成成功")
        print(f"  特征数: {len(input_columns)}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n" + "="*60)
    print("测试模型创建...")
    print("="*60)
    
    try:
        from models_pytorch import MFP
        import torch
        
        # 简单配置
        input_columns = {
            'type': {'is_sequence': True, 'type': 'categorical', 'input_dim': 6, 'shape': [1]},
            'left': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
            'top': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
            'width': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
            'height': {'is_sequence': True, 'type': 'categorical', 'input_dim': 64, 'shape': [1]},
        }
        
        model = MFP(input_columns, embed_dim=128, num_blocks=2, num_heads=4)
        print(f"✓ 模型创建成功")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  总参数: {total_params:,}")
        
        # 测试前向传播
        batch_size = 2
        seq_len = 10
        test_input = {
            'length': torch.tensor([[5], [7]], dtype=torch.long),
            'type': torch.randint(0, 6, (batch_size, seq_len, 1)),
            'left': torch.randint(0, 64, (batch_size, seq_len, 1)),
            'top': torch.randint(0, 64, (batch_size, seq_len, 1)),
            'width': torch.randint(0, 64, (batch_size, seq_len, 1)),
            'height': torch.randint(0, 64, (batch_size, seq_len, 1)),
        }
        
        with torch.no_grad():
            outputs = model(test_input)
        
        print(f"✓ 前向传播测试成功")
        print(f"  输出数: {len(outputs)}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_directories(save_dir, log_dir):
    """检查并创建必要目录"""
    print("\n" + "="*60)
    print("检查目录结构...")
    print("="*60)
    
    directories = {
        save_dir: 'Checkpoints目录',
        log_dir: 'Logs目录',
    }
    
    for dir_path, desc in directories.items():
        dir_path = Path(dir_path)
        if dir_path.exists():
            print(f"✓ {desc:20s} 存在: {dir_path}")
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ {desc:20s} 已创建: {dir_path}")


def main():
    print("\n" + "="*80)
    print("MFP 训练环境检查")
    print("="*80 + "\n")
    
    # 配置路径
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "crello_json"
    save_dir = project_root / "checkpoints"
    log_dir = project_root / "logs"
    
    # 运行检查
    checks = [
        ("依赖包", lambda: check_dependencies()),
        ("数据文件", lambda: check_data(data_dir)),
        ("目录结构", lambda: check_directories(save_dir, log_dir) or True),
        ("数据加载", lambda: test_data_loading(data_dir)),
        ("模型创建", lambda: test_model_creation()),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} 检查出错: {e}")
            results.append((name, False))
    
    # 总结
    print("\n" + "="*80)
    print("检查总结")
    print("="*80)
    
    all_passed = True
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:20s}: {status}")
        if not result:
            all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("\n🎉 所有检查通过！可以开始训练了！")
        print("\n快速开始:")
        print("  cd scripts")
        print("  ./train.sh")
        print("\n或者:")
        print("  python train_pytorch_improved.py --data_dir ../data/crello_json --config config/train_config.json")
    else:
        print("\n⚠ 部分检查失败，请解决问题后再开始训练")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
    
#!/usr/bin/env bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES="0"
export PYTHONUNBUFFERED="1"

# 定义Python解释器路径
PYTHON_PATH="/home/dell/anaconda3/envs/ptfd311/bin/python"

# 定义脚本和配置路径
SCRIPT_PATH="/home/dell/Project-HCL/BaseLine/flexdm_pt/scripts/train_pytorch_improved.py"
DATA_DIR="/storage/HCL_data/crello_original/to_json"
CONFIG_PATH="/home/dell/Project-HCL/BaseLine/flexdm_pt/scripts/config/train_config.json"
SAVE_DIR="/home/dell/Project-HCL/BaseLine/flexdm_pt/checkpoints/retrain"
LOG_DIR="/home/dell/Project-HCL/BaseLine/flexdm_pt/scripts/logs/retrain"

# 打印运行信息
echo "开始运行训练脚本..."
echo "使用Python解释器: $PYTHON_PATH"
echo "数据目录: $DATA_DIR"
echo "配置文件: $CONFIG_PATH"
echo "保存目录: $SAVE_DIR"
echo "日志目录: $LOG_DIR"

# 执行训练命令
$PYTHON_PATH $SCRIPT_PATH \
    --data_dir $DATA_DIR \
    --config $CONFIG_PATH \
    --device cuda \
    --save_dir $SAVE_DIR \
    --log_dir $LOG_DIR

# 检查命令执行结果
if [ $? -eq 0 ]; then
    echo "训练脚本执行完成"
else
    echo "训练脚本执行失败"
    exit 1
fi

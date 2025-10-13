#!/bin/bash
# 快速启动训练脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用哪个GPU，设置为空字符串使用CPU
export OMP_NUM_THREADS=4

# 项目路径
PROJECT_ROOT="/home/dell/Project-HCL/BaseLine/flexdm_pt"
DATA_DIR="${PROJECT_ROOT}/data/crello_json"
SAVE_DIR="${PROJECT_ROOT}/checkpoints"
LOG_DIR="${PROJECT_ROOT}/logs"
CONFIG="${PROJECT_ROOT}/scripts/config/train_config.json"

# 创建必要的目录
mkdir -p ${SAVE_DIR}
mkdir -p ${LOG_DIR}
mkdir -p ${PROJECT_ROOT}/scripts/config

# 训练参数
BATCH_SIZE=16
NUM_EPOCHS=100
LEARNING_RATE=0.0001
EMBED_DIM=256
NUM_BLOCKS=4
NUM_HEADS=8
NUM_WORKERS=4

# 检查数据是否存在
if [ ! -d "${DATA_DIR}" ]; then
    echo "错误: 数据目录不存在: ${DATA_DIR}"
    exit 1
fi

if [ ! -f "${DATA_DIR}/train.json" ]; then
    echo "错误: 训练数据不存在: ${DATA_DIR}/train.json"
    exit 1
fi

# 启动训练
echo "=================================="
echo "开始训练 MFP 模型"
echo "=================================="
echo "数据目录: ${DATA_DIR}"
echo "保存目录: ${SAVE_DIR}"
echo "日志目录: ${LOG_DIR}"
echo "批次大小: ${BATCH_SIZE}"
echo "训练轮数: ${NUM_EPOCHS}"
echo "=================================="
echo ""

cd ${PROJECT_ROOT}/scripts

python train_pytorch_improved.py \
    --data_dir ${DATA_DIR} \
    --config ${CONFIG} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --embed_dim ${EMBED_DIM} \
    --num_blocks ${NUM_BLOCKS} \
    --num_heads ${NUM_HEADS} \
    --num_workers ${NUM_WORKERS} \
    --save_dir ${SAVE_DIR} \
    --log_dir ${LOG_DIR} \
    --device cuda

echo ""
echo "=================================="
echo "训练完成！"
echo "模型保存在: ${SAVE_DIR}"
echo "日志保存在: ${LOG_DIR}"
echo "=================================="

# 训练完成后自动验证
echo ""
echo "开始验证模型..."
python validate.py \
    --data_dir ${DATA_DIR} \
    --checkpoint ${SAVE_DIR}/best.pth \
    --config ${CONFIG} \
    --split val \
    --device cuda

echo ""
echo "全部完成！"
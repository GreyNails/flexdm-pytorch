#!/bin/bash

################################################################################
# MFP Model Training Script
# 用法: bash train.sh [CONFIG] [GPU_ID]
# 示例: bash train.sh basic 0
################################################################################

set -e  # 遇到错误立即退出

# ==================== 配置区域 ====================

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
CONFIG_TYPE=${1:-"basic"}  # basic, debug, full
GPU_ID=${2:-"0"}

# 数据路径
DATA_DIR="./data/crello_json"
CONFIG_DIR="./config"

# 根据配置类型设置参数
case $CONFIG_TYPE in
    "basic")
        echo -e "${GREEN}[INFO]${NC} 使用基础训练配置"
        CONFIG_FILE="${CONFIG_DIR}/train_config.json"
        BATCH_SIZE=32
        NUM_EPOCHS=100
        LEARNING_RATE=1e-4
        SAVE_DIR="./checkpoints"
        LOG_DIR="./logs"
        ;;
    
    "debug")
        echo -e "${YELLOW}[INFO]${NC} 使用调试配置（小数据量）"
        CONFIG_FILE="${CONFIG_DIR}/train_config_debug.json"
        BATCH_SIZE=4
        NUM_EPOCHS=2
        LEARNING_RATE=1e-4
        SAVE_DIR="./checkpoints_debug"
        LOG_DIR="./logs_debug"
        ;;
    
    "full")
        echo -e "${BLUE}[INFO]${NC} 使用完整训练配置（长时间）"
        CONFIG_FILE="${CONFIG_DIR}/train_config.json"
        BATCH_SIZE=64
        NUM_EPOCHS=200
        LEARNING_RATE=5e-5
        SAVE_DIR="./checkpoints_full"
        LOG_DIR="./logs_full"
        ;;
    
    "resume")
        echo -e "${GREEN}[INFO]${NC} 恢复训练"
        CONFIG_FILE="${CONFIG_DIR}/train_config.json"
        BATCH_SIZE=32
        NUM_EPOCHS=100
        LEARNING_RATE=1e-4
        SAVE_DIR="./checkpoints"
        LOG_DIR="./logs"
        RESUME_PATH="${SAVE_DIR}/latest.pth"
        ;;
    
    *)
        echo -e "${RED}[ERROR]${NC} 未知配置类型: $CONFIG_TYPE"
        echo "可用配置: basic, debug, full, resume"
        exit 1
        ;;
esac

# ==================== 环境检查 ====================

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  MFP 模型训练${NC}"
echo -e "${BLUE}========================================${NC}\n"

# 检查Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python未安装"
    exit 1
fi
echo -e "${GREEN}[✓]${NC} Python版本: $(python --version)"

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}[✓]${NC} CUDA可用"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader -i $GPU_ID
else
    echo -e "${YELLOW}[WARNING]${NC} CUDA不可用，将使用CPU训练"
    GPU_ID="cpu"
fi

# 检查数据目录
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}[ERROR]${NC} 数据目录不存在: $DATA_DIR"
    echo "请先准备数据或修改DATA_DIR路径"
    exit 1
fi
echo -e "${GREEN}[✓]${NC} 数据目录: $DATA_DIR"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}[WARNING]${NC} 配置文件不存在: $CONFIG_FILE"
    echo "将使用命令行参数"
fi

# 创建必要的目录
mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"
echo -e "${GREEN}[✓]${NC} 输出目录已创建"

# 检查依赖包
echo -e "\n${BLUE}检查Python依赖...${NC}"
python -c "import torch; import numpy; import tqdm" 2>/dev/null || {
    echo -e "${RED}[ERROR]${NC} 缺少必要的Python包"
    echo "请运行: pip install torch numpy tqdm tensorboard"
    exit 1
}
echo -e "${GREEN}[✓]${NC} 依赖检查完成"

# ==================== 训练参数 ====================

echo -e "\n${BLUE}训练参数:${NC}"
echo "  配置类型: $CONFIG_TYPE"
echo "  GPU ID: $GPU_ID"
echo "  批次大小: $BATCH_SIZE"
echo "  训练轮数: $NUM_EPOCHS"
echo "  学习率: $LEARNING_RATE"
echo "  保存目录: $SAVE_DIR"
echo "  日志目录: $LOG_DIR"

if [ -n "$RESUME_PATH" ] && [ -f "$RESUME_PATH" ]; then
    echo "  恢复训练: $RESUME_PATH"
fi

# 确认继续
echo -e "\n${YELLOW}按Enter继续，Ctrl+C取消...${NC}"
read -r

# ==================== 开始训练 ====================

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  开始训练${NC}"
echo -e "${GREEN}========================================${NC}\n"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONUNBUFFERED=1

# 记录开始时间
START_TIME=$(date +%s)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

# 构建训练命令
TRAIN_CMD="python train_mfp_pytorch.py \
    --data_dir $DATA_DIR \
    --config $CONFIG_FILE \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --device $([ "$GPU_ID" == "cpu" ] && echo "cpu" || echo "cuda") \
    --save_dir $SAVE_DIR \
    --log_dir $LOG_DIR"

# 添加恢复训练参数
if [ -n "$RESUME_PATH" ] && [ -f "$RESUME_PATH" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME_PATH"
fi

# 执行训练（同时输出到终端和日志文件）
echo "训练命令: $TRAIN_CMD" | tee "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 使用trap捕获中断信号
trap 'echo -e "\n${RED}训练被中断${NC}"; exit 130' INT TERM

# 执行训练
eval $TRAIN_CMD 2>&1 | tee -a "$LOG_FILE"
TRAIN_STATUS=${PIPESTATUS[0]}

# ==================== 训练完成 ====================

# 计算训练时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "" | tee -a "$LOG_FILE"
echo -e "${GREEN}========================================${NC}" | tee -a "$LOG_FILE"

if [ $TRAIN_STATUS -eq 0 ]; then
    echo -e "${GREEN}  训练完成！${NC}" | tee -a "$LOG_FILE"
else
    echo -e "${RED}  训练失败！退出码: $TRAIN_STATUS${NC}" | tee -a "$LOG_FILE"
fi

echo -e "${GREEN}========================================${NC}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "训练时间: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "模型保存: $SAVE_DIR" | tee -a "$LOG_FILE"
echo "TensorBoard日志: $LOG_DIR" | tee -a "$LOG_FILE"

# ==================== 后处理 ====================

if [ $TRAIN_STATUS -eq 0 ]; then
    echo -e "\n${BLUE}可用操作:${NC}"
    echo "  查看TensorBoard: tensorboard --logdir=$LOG_DIR --port=6006"
    echo "  运行Demo: bash demo.sh pos"
    echo "  查看日志: cat $LOG_FILE"
    
    # 检查最佳模型
    if [ -f "${SAVE_DIR}/best.pth" ]; then
        BEST_SIZE=$(du -h "${SAVE_DIR}/best.pth" | cut -f1)
        echo -e "\n${GREEN}[✓]${NC} 最佳模型: ${SAVE_DIR}/best.pth (${BEST_SIZE})"
    fi
fi

exit $TRAIN_STATUS
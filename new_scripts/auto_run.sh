#!/bin/bash

# 目标 GPU 编号
GPU_ID=0

# 设定显存阈值（单位：MB）
THRESHOLD=10000

# 标志文件
FLAG_FILE="./run_sh_executed"

# 如果标志文件已存在，则不再运行
if [ -f "$FLAG_FILE" ]; then
    echo "run.sh has already been executed. Exiting..."
    exit 0
fi

# 获取 2 号显卡的显存使用情况（单位：MB）
MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((GPU_ID+1))p")

echo "GPU $GPU_ID Memory Used: $MEM_USED MB"

# 检查显存是否低于阈值
if [ "$MEM_USED" -lt "$THRESHOLD" ]; then
    echo "Memory below $THRESHOLD MB, running run.sh..."
    sh run.sh
    # 创建标志文件，防止后续再次执行
    touch "$FLAG_FILE"
else
    echo "Memory usage is above $THRESHOLD MB, not running run.sh."
fi
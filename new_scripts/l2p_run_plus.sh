#!/bin/bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=4

# 基础命令
BASE_COMMAND="python"  # 可以是nohup python 或 python

# 可选参数
########################################################################################
B_DATASET="medmcqa"                                              # 支持 "medmcqa" "sciq" "piqa"
C_DATASET="piqa"                                              # 支持 "medmcqa" "sciq" "piqa"
# D_DATASET="sciq"                                           # 支持 "medmcqa" "sciq" "piqa"           
MODEL_NAME="llama8B"                                             #可以选 "qwen","gptj","llama"                                              #正则化系数
########################################################################################

FISHER_MATRIX_PATH="weights/fisher-matrix/fisher-matrix-6B" # Fisher矩阵目录
PER_DEVICE_TRAIN_BATCH_SIZE=20                              # 每个设备的训练batchsize
GRADIENT_ACCUMULATION_STEPS=4
LR=8e-4                                                        
EWC_LAMBDA=0.5
LORA_RANK=8                                               # 秩的大小
A_NUM_EPOCHS=5                                                # 训练轮数
B_NUM_EPOCHS=3                                                # 训练轮数
C_NUM_EPOCHS=3                                                # 训练轮数
# D_NUM_EPOCHS=2                                                # 训练轮数

A_TRAIN_FILE="datasets/pile_test.jsonl"
A_TEMPLATE_NAME="pile"

cd ..

# 选择大模型
if [[ "$MODEL_NAME" == "qwen" ]]; then
    MODEL_NAME_OR_PATH="weights/Qwen2.5-7B-Instruct"
elif [[ "$MODEL_NAME" == "gptj" ]]; then
    MODEL_NAME_OR_PATH="weights/gpt-j-6b"
elif [[ "$MODEL_NAME" == "llama8B" ]]; then
    MODEL_NAME_OR_PATH="weights/Meta-Llama-3.1-8B-Instruct"
else
    echo "Invalid model. Please choose 'qwen', 'gptj', 'llama'."
    exit 1
fi

# 训练数据集选择
choose_dataset() {
    local DATASET=$1
    local TRAIN_FILE=""
    local TEMPLATE_NAME=""

    if [ "$DATASET" = "medmcqa" ]; then
        TRAIN_FILE="datasets/medmcqa/train.json"
        TEMPLATE_NAME="medmcqa"
    elif [ "$DATASET" = "sciq" ]; then
        TRAIN_FILE="datasets/sciq/train.json"
        TEMPLATE_NAME="sciq"
    elif [ "$DATASET" = "piqa" ]; then
        TRAIN_FILE="datasets/piqa/train_new.json"
        TEMPLATE_NAME="piqa"
    else
        echo "Invalid dataset: $DATASET. Please choose 'sciq', 'medmcqa', or 'piqa'."
        exit 1
    fi

    echo "$TRAIN_FILE,$TEMPLATE_NAME"
}

# 获取BCD的数据集文件和模板名称
B_RESULT=$(choose_dataset "$B_DATASET")
C_RESULT=$(choose_dataset "$C_DATASET")
# D_RESULT=$(choose_dataset "$D_DATASET")

# 将返回的结果分割为文件路径和模板名称
B_TRAIN_FILE=$(echo "$B_RESULT" | cut -d',' -f1)
B_TEMPLATE_NAME=$(echo "$B_RESULT" | cut -d',' -f2)

C_TRAIN_FILE=$(echo "$C_RESULT" | cut -d',' -f1)
C_TEMPLATE_NAME=$(echo "$C_RESULT" | cut -d',' -f2)

# D_TRAIN_FILE=$(echo "$D_RESULT" | cut -d',' -f1)
# D_TEMPLATE_NAME=$(echo "$D_RESULT" | cut -d',' -f2)

# 根据模型选择不同的脚本

SCRIPT="l2p_run_plus.py"
LOG_FILE="log/l2p_plus-$MODEL_NAME-r$LORA_RANK-C$REG_CONF-$B_DATASET->$C_DATASET.log"
OUTPUT_DIR="output/l2p_plus-$MODEL_NAME-r$LORA_RANK-C$REG_CONF-$B_DATASET->$C_DATASET/checkpoint"     


echo $LOG_FILE
echo $OUTPUT_DIR
echo $CUDA_VISIBLE_DEVICES

# 运行命令
nohup python $SCRIPT \
    --A_train_file "$A_TRAIN_FILE" \
    --B_train_file "$B_TRAIN_FILE" \
    --C_train_file "$C_TRAIN_FILE" \
    --fisher_matrix_path "$FISHER_MATRIX_PATH" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --lr $LR \
    --output_dir "$OUTPUT_DIR" \
    --ewc_lambda $EWC_LAMBDA \
    --A_num_epochs $A_NUM_EPOCHS \
    --B_num_epochs $B_NUM_EPOCHS \
    --C_num_epochs $C_NUM_EPOCHS \
    --A_template_name $A_TEMPLATE_NAME \
    --B_template_name $B_TEMPLATE_NAME \
    --C_template_name $C_TEMPLATE_NAME \
    --lora_r $LORA_RANK \
    --cuda $CUDA_VISIBLE_DEVICES \
    >> "$LOG_FILE" 2>&1 &

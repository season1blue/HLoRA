#!/bin/bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=2

# 基础命令
BASE_COMMAND="python"  # 可以是nohup python 或 python

# 可选参数
########################################################################################
DATASET="sciq"                                              # 支持 "medmcqa" "sciq" "piqa"           
MODEL_NAME="qwen"                                           #可以选 "qwen","gptj","llama"
REG_CONF=1e-3                                             #正则化系数
########################################################################################

FISHER_MATRIX_PATH="weights/fisher-matrix/fisher-matrix-6B" # Fisher矩阵目录
PER_DEVICE_TRAIN_BATCH_SIZE=20                              # 每个设备的训练batchsize
GRADIENT_ACCUMULATION_STEPS=4
LR=8e-4                                                        
EWC_LAMBDA=0.5
LORA_RANK=8                                               # 秩的大小
NUM_EPOCHS=5                                                # 训练轮数
A_TRAIN_FILE="datasets/pile_test.jsonl"
A_TEMPLATE_NAME="pile"

cd ..

# 选择大模型
if [[ "$MODEL_NAME" == "qwen" ]]; then
    MODEL_NAME_OR_PATH="weights/Qwen2.5-7B-Instruct"
elif [[ "$MODEL_NAME" == "gptj" ]]; then
    MODEL_NAME_OR_PATH="weights/gpt-j-6b"
elif [[ "$MODEL_NAME" == "llama" ]]; then
    MODEL_NAME_OR_PATH="weights/llama3-2-3b-instruct"
else
    echo "Invalid model. Please choose 'qwen', 'gptj', 'llama'."
    exit 1
fi

# 训练数据集选择
if [[ "$DATASET" == "medmcqa" ]]; then
    B_TRAIN_FILE="datasets/medmcqa/train.json"   
    B_TEMPLATE_NAME="medmcqa"
elif [[ "$DATASET" == "sciq" ]]; then
    B_TRAIN_FILE="datasets/sciq/train.json"   
    B_TEMPLATE_NAME="sciq"
elif [[ "$DATASET" == "piqa" ]]; then
    B_TRAIN_FILE="datasets/piqa/train_new.json"   
    B_TEMPLATE_NAME="piqa"
else
    echo "Invalid datasets. Please choose 'sciq', 'medmcqa', 'piqa'."
    exit 1
fi

# 根据模型选择不同的脚本
SCRIPT="SI_run_learning.py"
LOG_FILE="log/slora-$MODEL_NAME-$DATASET-r$LORA_RANK-C$REG_CONF-learning.log"
OUTPUT_DIR="output/slora-$MODEL_NAME-$DATASET-r$LORA_RANK-C$REG_CONF-learning/checkpoint"     


echo $LOG_FILE
echo $OUTPUT_DIR
echo $DATASET
echo $CUDA_VISIBLE_DEVICES

# 运行命令
nohup python $SCRIPT \
    --A_train_file "$A_TRAIN_FILE" \
    --B_train_file "$B_TRAIN_FILE" \
    --fisher_matrix_path "$FISHER_MATRIX_PATH" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --lr $LR \
    --output_dir "$OUTPUT_DIR" \
    --ewc_lambda $EWC_LAMBDA \
    --num_epochs $NUM_EPOCHS \
    --A_template_name $A_TEMPLATE_NAME \
    --B_template_name $B_TEMPLATE_NAME \
    --lora_r $LORA_RANK \
    --reg_conf $REG_CONF \
    --cuda $CUDA_VISIBLE_DEVICES \
    >> "$LOG_FILE" 2>&1 &

export CUDA_VISIBLE_DEVICES=0

# 基础命令
BASE_COMMAND="nohup python"  # 可以是nohup python 或 python

# 可选参数
########################################################################################
DATASET="medmcqa"                                              # 支持 "medmcqa" "sciq" "piqa"
MODEL="jslora"                                              # 可以修改为 "jslora" 或 "ewclora" 或 "rslora" 来运行对应的模型
FISHER_MATRIX_PATH="weights/fisher-matrix/fisher-pile-qwen" # Fisher矩阵目录
MODEL_NAME="llama"                                           #可以选 "qwen","gptj","llama"
########################################################################################

PER_DEVICE_TRAIN_BATCH_SIZE=20                              # 每个设备的训练batchsize
GRADIENT_ACCUMULATION_STEPS=4
LR=8e-4                                                        
EWC_LAMBDA=0.5
LORA_RANK=2                                                 # 秩的大小
NUM_EPOCHS=5                                                # 训练轮数

cd ..

# 选择大模型
if [[ "$MODEL_NAME" == "qwen" ]]; then
    MODEL_NAME_OR_PATH="weights/Qwen2.5-7B-Instruct"
elif [[ "$MODEL_NAME" == "gptj" ]]; then
    MODEL_NAME_OR_PATH="weights/gpt-j-6b"
elif [[ "$MODEL_NAME" == "llama" ]]; then
    MODEL_NAME_OR_PATH="weights/Llama-3.2-3B-Instruct"
else
    echo "Invalid model. Please choose 'qwen', 'gptj', 'llama'."
    exit 1
fi

# 训练数据集选择
if [[ "$DATASET" == "medmcqa" ]]; then
    TRAIN_FILE="datasets/medmcqa/train.json"   
    TEMPLATE_NAME="medmcqa"
elif [[ "$DATASET" == "sciq" ]]; then
    TRAIN_FILE="datasets/sciq/train.json"   
    TEMPLATE_NAME="sciq"
elif [[ "$DATASET" == "piqa" ]]; then
    TRAIN_FILE="datasets/piqa/train_new.json"   
    TEMPLATE_NAME="piqa"
else
    echo "Invalid datasets. Please choose 'sciq', 'medmcqa', 'piqa'."
    exit 1
fi

# 根据模型选择不同的脚本
if [[ "$MODEL" == "jslora" ]]; then
    SCRIPT="run.py"
    LOG_FILE="log/jslora-$MODEL_NAME-$DATASET-r$LORA_RANK.log"
    OUTPUT_DIR="output/jslora-$MODEL_NAME-$DATASET-r$LORA_RANK"     
elif [[ "$MODEL" == "ewclora" ]]; then
    SCRIPT="EWC_run.py"
    LOG_FILE="log/ewclora-$MODEL_NAME-$DATASET-r$LORA_RANK-ewc_conf=0.5.log"
    OUTPUT_DIR="output/ewclora-$MODEL_NAME-$DATASET-r$LORA_RANK-ewc_conf=0.5"
elif [[ "$MODEL" == "rslora" ]]; then
    SCRIPT="rslora_run.py"
    LOG_FILE="log/rslora-$MODEL_NAME-$DATASET-r$LORA_RANK.log"
    OUTPUT_DIR="output/rslora-$MODEL_NAME-$DATASET-r$LORA_RANK"             
else
    echo "Invalid model specified. Please choose 'lora', 'rslora', 'ewc-lora', or 'slora'."
    exit 1
fi

echo $LOG_FILE
echo $OUTPUT_DIR
echo $DATASET
echo $CUDA_VISIBLE_DEVICES

# 运行命令
$BASE_COMMAND $SCRIPT \
    --train_file "$TRAIN_FILE" \
    --fisher_matrix_path "$FISHER_MATRIX_PATH" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --lr $LR \
    --output_dir "$OUTPUT_DIR" \
    --ewc_lambda $EWC_LAMBDA \
    --num_epochs $NUM_EPOCHS \
    --template_name $TEMPLATE_NAME \
    --lora_r $LORA_RANK  \
    >> "$LOG_FILE" 2>&1 &
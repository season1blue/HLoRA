#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
 
# 模型路径映射
declare -A model_paths=(
    ["gptj"]="../weights/gpt-j-6b"
    ["gpt-neo"]="../weights/gpt-neo-1.3B"
    ["llama"]="../weights/Llama-2-7b-hf"
    ["meta-llama-3"]="../weights/Meta-Llama-3-8B"
    ["meta-llama-3.1"]="../weights/Meta-Llama-3.1-8B-Instruct"
    ["minicpm3"]="../weights/MiniCPM3-4B"
    ["llama-fft"]="../../lpframework/llamapro/output/piqa-llama2/"
)

# 设置参数
MODEL_IDENTIFIERS=("llama-fft")  # 模型标识符数组
LORA_METHODS=("rslora" )  # LoRA 方法 "none" "lora" "ewclora"
TASKS="piqa"  # 设置任务，例如"piqa" 或其他评估任务

for MODEL_IDENTIFIER in "${MODEL_IDENTIFIERS[@]}"
do
    model_dir="${model_paths[$MODEL_IDENTIFIER]}"  # 从映射中获取模型路径
    for LORA_METHOD in "${LORA_METHODS[@]}"
    do
        echo "\n\n ##### MODEL: $MODEL_IDENTIFIER  METHOD: $LORA_METHOD   ######"

        model_args="pretrained=${model_dir},trust_remote_code=True"
        if [ "$LORA_METHOD" != "none" ]; then
            lora_dir="../output/rightlabel-${LORA_METHOD}-${MODEL_IDENTIFIER}-${TASKS}/checkpoint"
            # model_args="${model_args},peft=${lora_dir}"
            model_args="${model_args}"
            echo "Using LoRA from: $lora_dir"
        fi

        # 执行评估命令
        gg lm_eval --model hf \
            --model_args "$model_args" \
            --tasks $TASKS \
            --batch_size auto:16 \
            --device auto \
            --output_path "./output/eval_out/fft"

        echo "Evaluation completed for $MODEL_IDENTIFIER with $LORA_METHOD"
    done
done

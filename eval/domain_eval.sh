#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
cd ..

MODEL_NAME_OR_PATH="weights/Meta-Llama-3.1-8B-Instruct"    # Meta-Llama-3.1-8B-Instruct    Llama-3.2-3B-Instruct
# lora_dir="output/slora-gptj-sciq-r8-C1e-3-w0-layerwise/checkpoint/epoch_5"
lora_dirS=(
"output/gem_plus-llama8B-r8-C-sciq->medmcqa/checkpoint/medmcqa_epoch_1"
"output/gem_plus-llama8B-r8-C-sciq->medmcqa/checkpoint/medmcqa_epoch_2"
"output/gem_plus-llama8B-r8-C-sciq->medmcqa/checkpoint/medmcqa_epoch_3"
"output/gem_plus-llama8B-r8-C-sciq->medmcqa/checkpoint/sciq_epoch_1"
"output/gem_plus-llama8B-r8-C-sciq->medmcqa/checkpoint/sciq_epoch_2"
"output/gem_plus-llama8B-r8-C-sciq->medmcqa/checkpoint/sciq_epoch_3"
)
# 设置参数
MODEL_IDENTIFIER="gptj"  # 模型标识符数组
LORA_METHOD="slora"  # LoRA 方法 "none" "lora" "ewclora"
TASKS="sciq"  # 设置任务，例如"piqa" 或其他评估任务 medmcqa

echo "\n\n ##### MODEL: $MODEL_IDENTIFIER  METHOD: $LORA_METHOD   ######"


# model_args="${model_args}"

for lora_dir in "${lora_dirS[@]}"; do
    model_args="pretrained=${MODEL_NAME_OR_PATH},trust_remote_code=True"
    model_args="${model_args},peft=${lora_dir}"
    # 执行评估命令
    lm_eval --model hf \
        --model_args "$model_args" \
        --tasks $TASKS \
        --batch_size auto:16 \
        --device auto \
        --output_path "$lora_dir"

    echo "Evaluation completed for $MODEL_IDENTIFIER with $LORA_METHOD"
done

#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
cd ..
# 设置参数
LORA_METHODS="jslora"  # LoRA 方法数组，包含 "none" 表示不使用 LoRA
BATCH_SIZE=20  # 批处理大小
VAL_FILE="datasets/pile_test.jsonl"
# LORA_METHOD= "lora"
MODEL_NAME_OR_PATH="weights/Llama-3.2-3B-Instruct"     # Meta-Llama-3.1-8B-Instruct    Llama-3.2-3B-Instruct
# LORA_PATH="output/slora-llama-sciq-r8-C=1e-3-w=0"  # 可以为"none"
# LORA_PATH="none"


LORA_PATHS=(
"output/l2p-llama-sciq-r8"
# "output/gem_plus-llama8B-r8-C-sciq->medmcqa/checkpoint/medmcqa_epoch_2"
# "output/gem_plus-llama8B-r8-C-sciq->medmcqa/checkpoint/medmcqa_epoch_3"
# "output/gem_plus-llama8B-r8-C-sciq->medmcqa/checkpoint/sciq_epoch_1"
# "output/gem_plus-llama8B-r8-C-sciq->medmcqa/checkpoint/sciq_epoch_2"
# "output/gem_plus-llama8B-r8-C-sciq->medmcqa/checkpoint/sciq_epoch_3"
# "output/headlora-llama-sciq-bs6-topk14/checkpoint/epoch_2"
# "output/headlora-llama-sciq-bs6-topk14/checkpoint/epoch_3"



# "output/rslora-llama-piqa-r8/epoch_2"
# "output/rslora-llama-medmcqa-r8"
)  # 每个路径写在新的一行

for LORA_PATH in "${LORA_PATHS[@]}"; do
    echo "Evaluating with LoRA path: $LORA_PATH"
    python eval/eval_ppl_pile.py \
        --model_identifier $MODEL_NAME_OR_PATH \
        --lora_name_or_path "$LORA_PATH" \
        --val_file $VAL_FILE \
        --per_device_eval_batch_size $BATCH_SIZE \
        --output_log_path "eval/result_general.log"
done


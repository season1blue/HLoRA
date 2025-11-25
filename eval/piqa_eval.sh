cd ..
export CUDA_VISIBLE_DEVICES=1

LORA_PATHS=(
# "output/gem_plus-llama8B-r8-C-sciq->piqa/checkpoint/sciq_epoch_1"
# "output/gem_plus-llama8B-r8-C-sciq->piqa/checkpoint/sciq_epoch_2"
# "output/gem_plus-llama8B-r8-C-sciq->piqa/checkpoint/sciq_epoch_3"
# "output/gem_plus-llama8B-r8-C-sciq->piqa/checkpoint/piqa_epoch_1"
"output/gem_plus-llama8B-r8-C-sciq->piqa/checkpoint/piqa_epoch_2"
"output/gem_plus-llama8B-r8-C-sciq->piqa/checkpoint/piqa_epoch_3"

)  # 每个路径写在新的一行
# Meta-Llama-3.1-8B-Instruct    Llama-3.2-3B-Instruct
for LORA_PATH in "${LORA_PATHS[@]}"; do
    echo "Evaluating with LoRA path: $LORA_PATH"
    python eval/eval_piqa.py \
        --data_path datasets/piqa/dev.jsonl \
        --labels_path datasets/piqa/train-labels.lst \
        --model_name_or_path weights/Meta-Llama-3.1-8B-Instruct \
        --lora_name_or_path $LORA_PATH \
        --load_lora 1 \
        --output_log \
        --output_path "eval/result_piqa.log"
done


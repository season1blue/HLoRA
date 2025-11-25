cd ..
export CUDA_VISIBLE_DEVICES=0

# LORA_PATH="output/slora-llama-medmcqa-r8-C=1e-3-w=0/checkpoint/epoch_1"

# python eval/eval_medmcqa_gen.py \
#     --model_name_or_path weights/llama3-2-3b-instruct \
#     --lora_name_or_path $LORA_PATH \
#     --val_file datasets/medmcqa/val.json \
#     --output_path "eval/output/eval_out/eval_gptj_lora_medmcqa"

LORA_PATHS=(
"output/gem_plus-llama8B-r8-C-sciq->medmcqa/checkpoint/medmcqa_epoch_1"
"output/gem_plus-llama8B-r8-C-sciq->medmcqa/checkpoint/medmcqa_epoch_2"
"output/gem_plus-llama8B-r8-C-sciq->medmcqa/checkpoint/medmcqa_epoch_3"
"output/gem_plus-llama8B-r8-C-sciq->medmcqa/checkpoint/sciq_epoch_1"
"output/gem_plus-llama8B-r8-C-sciq->medmcqa/checkpoint/sciq_epoch_2"
"output/gem_plus-llama8B-r8-C-sciq->medmcqa/checkpoint/sciq_epoch_3"
)  # 每个路径写在新的一行
# Meta-Llama-3.1-8B-Instruct    Llama-3.2-3B-Instruct
for LORA_PATH in "${LORA_PATHS[@]}"; do
    echo "Evaluating with LoRA path: $LORA_PATH"
    python eval/eval_medmcqa_gen.py \
        --model_name_or_path weights/Meta-Llama-3.1-8B-Instruct \
        --lora_name_or_path $LORA_PATH \
        --val_file datasets/medmcqa/val.json \
        --output_log_path "eval/result_medmcqa.log"
done


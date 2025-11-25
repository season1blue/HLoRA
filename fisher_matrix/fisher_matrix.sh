export CUDA_VISIBLE_DEVICES=3

cd ..

# 记录开始时间
echo "\n==== Job started at $(date) ====" >> fisher_matrix/compute_fisher.log

python fisher_matrix/compute_fisher_matrix.py \
    --dataset_name "pile" \
    --model_name_or_path "weights/llama3-2-3b-instruct" \
    --save_dir "weights/fisher-matrix/fisher-pile-llama" \
    --batch_size 8 \
    --data_size 5000  
    # \
    # >> fisher_matrix/compute_fisher_llama.log 2>&1 &


# nohup python fisher_matrix/compute_fisher_matrix.py \
#     --data_path "data/alpaca_data_cleaned.json" \
#     --model_name_or_path "weights/Llama-2-7b-hf" \
#     --save_dir "fisher_matrix/llama_weight" \
#     --size 20000 \
#     --resume \ 
#     >> fisher_matrix/compute_fisher.log 2>&1 &

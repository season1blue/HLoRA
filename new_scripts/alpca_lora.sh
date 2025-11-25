export CUDA_VISIBLE_DEVICES=0

TASK=medmcqa # gsm8k medmcqa pubmedqa
DATASET=openlifescienceai/medmcqa # openai/gsm8k  openlifescienceai/medmcqa qiaojin/PubMedQA
MODEL=Llama-2-7b-hf # gpt-j-6b or Llama-2-7b-hf

METHOD=lora #lora or rslora
LORA_R=8
EPOCH=8

cd ..

gg python si_alpca_$METHOD.py \
      --data_path $DATASET \
      --lora_r $LORA_R \
      --base_model weights/$MODEL\
      --num_epochs $EPOCH \
      --learning_rate 1e-4 \
      --batch_size 64 \
      --output_dir output/$MODEL-$METHOD-$TASK-r$LORA_R \
      --wandb_project llama_tune \
      --prompt_template_name $TASK
      # \
      # >> "log/$MODEL-$METHOD-$TASK-r$LORA_R.log" 2>&1 &
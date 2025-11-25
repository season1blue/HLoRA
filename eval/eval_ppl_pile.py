import argparse
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
import math
import os
import logging
from tqdm import tqdm

# 设置日志
def setup_logging():
    logging.basicConfig(filename='log_general.log', level=logging.INFO, 
                        format='%(asctime)s:%(levelname)s:%(message)s')

def main(args):
    setup_logging()
    
    # 指定 GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.specific_gpu
    accelerator = Accelerator()

    # 映射 model_identifier 到具体的模型路径
    model_paths = {
        "gptj": "../weights/gpt-j-6b",
        "gpt-neo": "../weights/gpt-neo-1.3B",
        "llama": "../weights/Llama-2-7b-hf",
        "meta-llama-3": "../weights/Meta-Llama-3-8B",
        "meta-llama-3.1": "../weights/Meta-Llama-3.1-8B-Instruct",
        "minicpm3": "../weights/MiniCPM3-4B"
    }

    model_name_or_path = args.model_identifier
    if not model_name_or_path:
        raise ValueError(f"Invalid model identifier. Provided: {args.model_identifier}, available: {list(model_paths.keys())}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token
    eval_dataset = load_dataset('json', data_files=args.val_file, split='train').select(range(5000))
    # eval_dataset = load_dataset('json', data_files=args.val_file, split='train')
    

    
    def preprocess_function(examples):
        batch = tokenizer(
            examples[args.text_column],
            max_length=args.max_src_len,
            padding='max_length',
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt',
        )
        batch['labels'] = batch['input_ids'].clone()
        batch['labels'][batch['labels'] == tokenizer.pad_token_id] = -100
        batch['seq_len'] = batch['attention_mask'].sum(1) - 1

        return batch

    with accelerator.main_process_first():
        eval_processed_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()

    def data_collator_longest_padding(features):
        batch = default_data_collator(features)
        max_len = batch['attention_mask'].sum(1).max()
        for k, v in batch.items():
            if k == 'seq_len':
                continue
            batch[k] = v[:, :max_len]
        return batch

    eval_dataloader = DataLoader(
        eval_processed_dataset, collate_fn=data_collator_longest_padding, batch_size=args.per_device_eval_batch_size, pin_memory=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_8bit=True, device_map={"": accelerator.local_process_index})
    
    # 根据 lora_name_or_path 判断是否加载 LoRA 微调模型
    if args.lora_name_or_path != "none":
        from peft import PeftModel  # 确保在这里导入 PeftModel，避免未使用LoRA时的不必要依赖
        model = PeftModel.from_pretrained(model, args.lora_name_or_path, device_map={"": accelerator.local_process_index})
        logging.info(f'Using LoRA model from: {args.lora_name_or_path}')
    else:
        logging.info(f'Using base model: {args.model_identifier}')

    model.eval()
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    accelerator.print(model)
    

    total_loss = 0
    total_seq_len = 0
    for _, batch in enumerate(tqdm(eval_dataloader, disable=not accelerator.is_local_main_process)):
        seq_len = batch.pop('seq_len').sum()
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss.double()
        loss *= seq_len
        loss = accelerator.gather(loss)
        seq_len = accelerator.gather(seq_len)
        total_loss += loss.sum().cpu().item()
        total_seq_len += seq_len.sum().cpu().item()

    ppl = math.pow(2, total_loss / total_seq_len) if total_seq_len > 0 else float('inf')
    logging.info(f'Model: {args.lora_name_or_path}, PPL: {ppl:.4f}')
    logging.info("-------")
    print(f'Evaluated {args.lora_name_or_path}, PPL: {ppl:.4f}')
    if accelerator.is_local_main_process:
        with open(args.output_log_path, "a") as f:
            f.write(f"{args.lora_name_or_path} evaluation PPL: {ppl:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the perplexity of a language model.")
    parser.add_argument('--model_identifier', type=str, required=True)
    parser.add_argument('--lora_name_or_path', type=str, required=True)
    parser.add_argument('--output_log_path', type=str, required=True)
    parser.add_argument('--val_file', type=str, required=True)
    parser.add_argument('--text_column', type=str, default='text')
    parser.add_argument('--per_device_eval_batch_size', type=int, required=True)
    parser.add_argument('--max_src_len', type=int, default=800)
    args = parser.parse_args()

    main(args)

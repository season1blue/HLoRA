import os
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)

from peft import PeftModel
import fire
import evaluate

def main(
    model_name_or_path="EleutherAI/gpt-j-6B",
    load_lora=True,
    lora_name_or_path="lora",
    val_file="medqa_val.json",
    text_column="question",
    label_column="cop",
    per_device_eval_batch_size=4,
    max_src_len=800,
    max_tgt_len=16,
    num_beams=4,
    sample_size=2000,
    output_log_path='eval/result_medmcqa.log',
):
    
    accelerator = Accelerator()

    eval_dataset = load_dataset(val_file.split(".")[-1], data_files={'validation': val_file})['validation']

    # if len(eval_dataset) > sample_size:
    #     eval_dataset = eval_dataset.shuffle(seed=42).select(range(sample_size))

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        map_dic = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        prompts = [
            f"Question: {question}\nChoices:\nA. {opa}\nB. {opb}\nC. {opc}\nD. {opd}\nAnswer:\n"
            for question, opa, opb, opc, opd in zip(
                examples[text_column], examples['opa'], examples['opb'], examples['opc'], examples['opd']
            )
        ]
        
        inputs = tokenizer(
            prompts,
            max_length=max_src_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        inputs['labels'] = examples[label_column]
        return inputs

    with accelerator.main_process_first():
        eval_processed_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,  # 并行处理
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()

    eval_dataloader = DataLoader(
        eval_processed_dataset, collate_fn=default_data_collator, batch_size=per_device_eval_batch_size, pin_memory=True
    )

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_8bit=True, device_map={"": accelerator.local_process_index})
    if load_lora:   
        model = PeftModel.from_pretrained(model, lora_name_or_path, device_map={"": accelerator.local_process_index})
    model.eval()

    gen_kwargs = {
        'max_new_tokens': max_tgt_len, 
        'num_beams': num_beams,
        'pad_token_id': tokenizer.eos_token_id,
    }

    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    eval_preds = []
    eval_labels = []
    for batch in tqdm(eval_dataloader, disable=not accelerator.is_local_main_process):
        labels = batch.pop('labels').detach().cpu().numpy()
        eval_labels.extend(labels)

        with torch.no_grad():
            outputs = accelerator.unwrap_model(model).generate(**batch, **gen_kwargs)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        eval_preds.extend(preds)

    # Post-process predictions
    def extract_answer(pred):
        # 找到 "Answer:" 后的部分
        if "Answer:" in pred:
            answer_start = pred.index("Answer:") + len("Answer:")
            # 去掉前后的空格并返回答案
            return pred[answer_start:].strip().split(".")[0]
        return ""

    eval_choice = [extract_answer(pred) for pred in eval_preds]

    # for i, pred in enumerate(eval_choice):
    #     print(f"Predicted choice: {pred}")
    #     print(f"Raw prediction: {eval_preds[i]}")
    #     print("---")


    # Calculate accuracy
    map_dic = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    eval_labels = [map_dic[label] for label in eval_labels]
    accuracy = sum(p == l for p, l in zip(eval_choice, eval_labels)) / len(eval_labels)

    output_str = f"Accuracy: {accuracy:.4f}"
    accelerator.print(f"Evaluation: {output_str}")

    if accelerator.is_local_main_process:
        with open(output_log_path, "a") as f:
            f.write(f"{lora_name_or_path} evaluation acc: {accuracy}\n")

    accelerator.wait_for_everyone()

if __name__ == "__main__":
    fire.Fire(main)

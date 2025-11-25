'''
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
'''
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig,LlamaForCausalLM,AutoModelForCausalLM,AutoTokenizer

import os
import sys
import argparse
from typing import List
from pathlib import Path

import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Subset

from LLMPruner.peft import (
    # LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from peft import LoraConfig
from LLMPruner.utils.prompter import Prompter, ZeroPrompter
from LLMPruner.datasets.ppl_dataset import get_loaders

device = "cuda" if torch.cuda.is_available() else "cpu"
from utils.regularization import SI

def main(args):
    # Set WanDB
    os.environ["WANDB_PROJECT"] = args.wandb_project

    # Load Pruned Model
    # pruned_dict = torch.load(args.prune_model, map_location='cpu')
    # tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # use_rslora=True
    )
    
    model = SI(args.base_model, args.learning_rate, config)
    
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.base_model,
    #     low_cpu_mem_usage=True if args.torch_version >=1.9 else False
    # )

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    if not args.no_instruction:
        prompter = Prompter(args.prompt_template_name)
    else:
        prompter = ZeroPrompter()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if device == 'cuda':
        model.half()

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        if 'lamini' in args.data_path.lower():
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                None,
                data_point["response"],
            )
        elif 'alpaca' in args.data_path.lower():
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"],
            )
        elif 'pubmedqa' in args.data_path.lower():
            full_prompt = prompter.generate_pubmedqa_prompt(
                data_point["question"],
                data_point["context"]['contexts'][0],
                data_point["final_decision"],
            )
        elif 'gsm8k' in args.data_path.lower():
            full_prompt = prompter.generate_gsm8k_prompt(
                data_point["question"],
                data_point["answer"],
            )
        elif 'medmcqa' in args.data_path.lower():
            full_prompt = prompter.generate_medmcqa_prompt(
                data_point["question"],
                data_point["opa"],
                data_point["opb"],
                data_point["opc"],
                data_point["opd"],
                data_point["cop"],
            )
        else:
            raise NotImplementedError

        tokenized_full_prompt = tokenize(full_prompt)
        if not args.train_on_inputs:
            if 'sciq' in args.data_path.lower():
                user_prompt = prompter.generate_prompt(
                    data_point["instruction"], data_point["input"] if 'input' in data_point.keys() else None,
                )
            if 'pubmedqa' in args.data_path.lower():
                user_prompt = prompter.generate_pubmedqa_prompt(
                    data_point["question"], data_point["context"]['contexts'][0] if 'context' in data_point.keys() else None,
                )
            if 'gsm8k' in args.data_path.lower():
                user_prompt = prompter.generate_gsm8k_prompt(
                    data_point["question"]
                )
            if 'medmcqa' in args.data_path.lower():
                user_prompt = prompter.generate_medmcqa_prompt(
                    data_point["question"],data_point["opa"],data_point["opb"],data_point["opc"],data_point["opd"]
                )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=args.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if args.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]  # could be sped up, probably
        return tokenized_full_prompt
    
    def direct_tokenize(data_point):

        tokenized_full_prompt = tokenize(full_prompt)
        if not args.train_on_inputs:
            if 'sciq' in args.data_path.lower():
                user_prompt = prompter.generate_prompt(
                    data_point["instruction"], data_point["input"] if 'input' in data_point.keys() else None,
                )
            if 'pubmedqa' in args.data_path.lower():
                user_prompt = prompter.generate_pubmedqa_prompt(
                    data_point["question"], data_point["context"]['contexts'][0] if 'context' in data_point.keys() else None,
                )
            if 'gsm8k' in args.data_path.lower():
                user_prompt = prompter.generate_gsm8k_prompt(
                    data_point["question"]
                )
            if 'medmcqa' in args.data_path.lower():
                user_prompt = prompter.generate_medmcqa_prompt(
                    data_point["question"],data_point["opa"],data_point["opb"],data_point["opc"],data_point["opd"]
                )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=args.add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if args.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    def split_and_tokenizer(test_data, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(test_data[field_name]), return_tensors='pt').input_ids[0]
        nsamples = test_ids.numel() // seq_len

        test_set = []
        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_set.append({
                'input_ids': batch,
                'labels': batch
            })
        return test_set

    # Prepare For LoRA
    model = prepare_model_for_int8_training(model)
    # model = get_peft_model(model, config)

    # Load Train Dataset for B Domain
    data = load_dataset(args.data_path, 'pqa_artificial')

    if args.cache_dataset and os.path.exists('datasets/cache/{}.bin'.format(args.data_path)):
        preprocess_data = torch.load('datasets/cache/{}.bin'.format(args.data_path))
        train_data, val_data = preprocess_data['train'], preprocess_data['val']
    else:
        train_val = data["train"].train_test_split(test_size=args.val_set_size, shuffle=True, seed=42)
        train_data = (train_val["train"].shuffle().map(generate_and_tokenize_prompt))
        val_data = {args.data_path: train_val["test"].shuffle().map(generate_and_tokenize_prompt),}

        if args.cache_dataset and args.local_rank == 0:
            cache_file = 'datasets/cache/{}.bin'.format(args.data_path)
            cache_dir = '/'.join(cache_file.split('/')[:-1])
            directory = Path(cache_dir)
            directory.mkdir(parents=True, exist_ok=True)

            torch.save({'train': train_data, 'val': val_data}, cache_file)

    general_data = load_dataset('monology/pile-uncopyrighted', "default")
    # Load Train Dataset for A Domain
    if args.cache_dataset and os.path.exists('datasets/cache/{}.bin'.format(args.data_path)):
        preprocess_data = torch.load('datasets/cache/{}.bin'.format(args.data_path))
        train_data, val_data = preprocess_data['train'], preprocess_data['val']
    else:
        train_val = general_data["train"].train_test_split(test_size=args.val_set_size, shuffle=True, seed=42)
        train_data = (train_val["train"].shuffle().map(generate_and_tokenize_prompt))
        val_data = {args.data_path: train_val["test"].shuffle().map(generate_and_tokenize_prompt),}

        if args.cache_dataset and args.local_rank == 0:
            cache_file = 'datasets/cache/{}.bin'.format(args.data_path)
            cache_dir = '/'.join(cache_file.split('/')[:-1])
            directory = Path(cache_dir)
            directory.mkdir(parents=True, exist_ok=True)

            torch.save({'train': train_data, 'val': val_data}, cache_file)
            
            
    #2w条数据
    if len(A_data) > 5000:
        indices = list(range(5000))
        A_data =Subset(A_data, indices)

    if len(B_data) > 20000:
        indices = list(range(20000))
        B_data =Subset(B_data, indices)

    A_data = general_data
    B_data = train_data

    # ------------------------------------------------------------------------------------------------

    from utils.SI import CustomTrainer

    print("Start Task A Training")
    trainer_a = CustomTrainer(
        model=model,
        train_dataset=A_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,  #  1
            learning_rate=args.learning_rate,
            bf16=True,
            logging_steps=10,
            logging_first_step=True,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            save_steps=200,
            output_dir=args.output_dir,
            save_total_limit=20,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=None,
            group_by_length=args.group_by_length,
            report_to="none",
            run_name=args.output_dir.split('/')[-1],
            metric_for_best_model="{}_loss".format(args.data_path),
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    trainer_a.set_task("A")
    
    print("Start Task B Training")
    trainer_b = CustomTrainer(
        model=model,
        train_dataset=B_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            bf16=True,
            logging_steps=10,
            logging_first_step=True,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            save_steps=200,
            output_dir=args.output_dir,
            save_total_limit=20,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=None,
            group_by_length=args.group_by_length,
            report_to="none",
            run_name=args.output_dir.split('/')[-1],
            metric_for_best_model="{}_loss".format(args.data_path),
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    trainer_b.set_task("B")
    
    
    
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    # trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.calculate_importance(task="None") ## 初始化重要性矩阵和记录prev_params
    trainer_a.train()
    model.calculate_importance(task="A") ## 计算A任务的重要性矩阵
    trainer_b.train()

    model.state_dict = old_state_dict
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--prune_model', type=str, help='prune model name')
    parser.add_argument('--data_path', type=str, default="yahma/alpaca-cleaned", help='data path')
    parser.add_argument('--cache_dataset', action="store_true", default=False)
    parser.add_argument('--extra_val_dataset', type=str, default=None, help='validation datasets. Split with ","')
    parser.add_argument('--output_dir', type=str, default="./lora-alpaca", help='output directory')

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--cutoff_len', type=int, default=256, help='cutoff length')
    parser.add_argument('--val_set_size', type=int, default=2000, help='validation set size')
    parser.add_argument('--prompt_template_name', type=str, default="alpaca", help="The prompt template to use, will default to alpaca.")
    parser.add_argument('--no_instruction', action='store_true', default=False, help="Whether to use the instruction template or not.")

    # Lora Configuration
    parser.add_argument('--lora_r', type=int, default=8, help='lora r')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--lora_target_modules', type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj", help='lora target modules')

    # llm hyperparameters
    parser.add_argument('--train_on_inputs', default=False, action="store_true", help='Train on inputs. If False, masks out inputs in loss')
    parser.add_argument('--add_eos_token', default=False, action="store_true")
    parser.add_argument('--group_by_length', default=False, action="store_true", help="faster, but produces an odd training loss curve")
   
    # wandb params
    parser.add_argument('--wandb_project', type=str, default="")
    parser.add_argument('--resume_from_checkpoint', type=str, help="either training checkpoint or final adapter")

    #ddp
    parser.add_argument('--local_rank', type=int, default=-1)
   
    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version

    main(args)

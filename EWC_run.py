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
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader,Subset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)
import time
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from peft.utils import transpose
import fire
from functools import partial
import bitsandbytes as bnb
from peft import prepare_model_for_int8_training
import math
import wandb
import numpy as np


class EWCLoRAModel(torch.nn.Module):
    def __init__(self, model_name_or_path, fisher_matrix_path, accelerator, ewc_lambda=1):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_8bit=True, device_map={"": accelerator.local_process_index})
        self.model = prepare_model_for_int8_training(self.model)

        self.fisher_matrix = AutoModelForCausalLM.from_pretrained(fisher_matrix_path, load_in_8bit=True, device_map={"": accelerator.local_process_index})
        self.fisher_matrix.eval()
        self.fisher_matrix.requires_grad_(False)

        self.ewc_lambda = ewc_lambda

    def get_peft_model(self, peft_config):
        self.model = get_peft_model(self.model, peft_config)

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def forward(self, **kwargs):
        labels = kwargs.pop("labels")
        # label_weights = kwargs.pop("label_weights")
        outputs = self.model(**kwargs)
        logits = outputs.logits
        ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        ce_loss = ce_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        ce_loss = torch.mean(ce_loss)

        # EWC loss
        fisher_matrix_module_dict = {name: module for name, module in self.fisher_matrix.named_modules()}
        ewc_loss = 0
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLayer):
                if module.active_adapter not in module.lora_A.keys():
                    continue
                if isinstance(module, bnb.nn.Linear8bitLt):
                    fan_in_fan_out=False
                else:
                    fan_in_fan_out = module.fan_in_fan_out
                adapter_weights = transpose(
                    module.lora_B[module.active_adapter].weight @ module.lora_A[module.active_adapter].weight,
                    fan_in_fan_out,
                ) * module.scaling[module.active_adapter]

                name = name.replace('base_model.model.', '')
                fisher_matrix_weights = fisher_matrix_module_dict[name].weight
                ewc_loss += torch.sum(fisher_matrix_weights * (adapter_weights ** 2))

        loss = ce_loss + self.ewc_lambda * ewc_loss
        # loss = ce_loss
        outputs.loss = loss
        outputs.ce_loss = ce_loss
        outputs.ewc_loss = ewc_loss
        return outputs
    
    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def save_pretrained(self, *args, **kwargs):
        self.model.save_pretrained(*args, **kwargs)

    def train(self, mode=True):
        self.model.train(mode)
    
    def eval(self):
        self.model.eval()



def main(
    model_name_or_path="EleutherAI/gpt-neo-1.3B",
    fisher_matrix_path="fisher_matrix.pt",
    train_file="train.json",
    lr=1e-3,
    num_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    seed=42,
    max_src_len=800,
    max_tgt_len=256,
    ewc_lambda=1,
    num_beams=1,
    output_dir="output",
    lora_r=8,
    lora_alpha=32,
    use_wandb=False,
    template_name="medmcqa"
):
        
    kwargs = DistributedDataParallelKwargs(static_graph=True)
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, kwargs_handlers=[kwargs])
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=lora_r, 
        lora_alpha=lora_alpha, 
        lora_dropout=0.1,
        target_modules=['v_proj', 'q_proj']
    )
    set_seed(seed)

    train_dataset = load_dataset("json", data_files={'train': train_file})['train']

    if use_wandb and accelerator.is_main_process:
        wandb.init(project='ewc-lora', config=wandb_args, save_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples, is_train=True):
        model_input = ''
        if template_name =="sciq":
            model_input = [
                f'Support: {support}\nQuestion: {question}\nAnswer: {correct_answer}'
                for support, question, correct_answer in zip(examples['support'], examples['question'], examples['correct_answer'])
            ]
        elif template_name =="medmcqa":
            map_dic = {0:'A',1:'B',2:'C',3:'D'}
            query = [
                f'Question:{question}\nChoices:\nA. {opa}\nB. {opb}\nC. {opc}\nD. {opd}\nAnswer:\n{map_dic[cop]}. '
                for question, opa, opb, opc, opd, cop in zip(examples['question'], examples['opa'], examples['opb'], examples['opc'], examples['opd'],examples['cop'])
            ]
            answer = []
            for i, iter in enumerate(examples['cop']):
                if iter == 0:
                    answer.append(examples['opa'][i])
                if iter == 1:
                    answer.append(examples['opb'][i])
                if iter == 2:
                    answer.append(examples['opc'][i])
                if iter == 3:
                    answer.append(examples['opd'][i])
            model_input = [x + y for x, y in zip(query, answer)]
        elif template_name =="piqa":
            # model_input = [
            #     f'Question: {goal}\nSolution1. {sol1}\nSolution2. {sol2}\nAnswer: {answer}'
            #     for goal, sol1, sol2, answer in zip(examples['goal'], examples['sol1'], examples['sol2'], examples['answer'])
            # ]
            model_input = examples['input']
            # # "input": "When boiling butter, when it's ready, you can: solution1. Pour it onto a plate, solution2. Pour it into a jar. Answer: solution2. Pour it into a jar"

        batch = tokenizer(
            model_input,
            max_length=max_src_len,
            padding='max_length',
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt',
        )

        if is_train:
            # prefix_weights = tokenizer(
            #     examples[text_column],
            #     max_length=max_src_len,
            #     padding='max_length',
            #     truncation=True,
            #     add_special_tokens=False,
            #     return_tensors='pt',
            # ).attention_mask[:, 1:]

            batch['labels'] = batch['input_ids'][:, 1:]
            batch['input_ids'] = batch['input_ids'][:, :-1]
            batch['attention_mask'] = batch['attention_mask'][:, 1:]
            # batch['label_weights'] = batch['attention_mask'] * (1 - prefix_weights).float()
            # if 'weight' in examples:
            #     batch['label_weights'] *= torch.tensor(examples['weight'])[:, None]

        return batch

    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            partial(preprocess_function, is_train=True),
            batched=True,
            num_proc=1,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()

    data_total_size = len(train_dataset)
    if data_total_size > 20000:
        indices = np.random.choice(data_total_size, 20000, replace=False)
        train_dataset =Subset(train_dataset,indices)
    

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=per_device_train_batch_size, pin_memory=True
    )

    #print(next(iter(train_dataloader)))

    # creating model
    model = EWCLoRAModel(model_name_or_path, fisher_matrix_path, accelerator, ewc_lambda=ewc_lambda)
    model.get_peft_model(peft_config)
    model.print_trainable_parameters()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )


    model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, optimizer, lr_scheduler
    )
    accelerator.print(model)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        start_time = time.time()
        total_steps = len(train_dataloader)
        progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process, ncols=200)

        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                loss_num = loss.detach().cpu().item()
                total_loss += loss_num
                ce_loss = outputs.ce_loss.detach().cpu().item()
                ewc_loss = outputs.ewc_loss.detach().cpu().item()
                
                # 更新 tqdm 描述信息
                progress_bar.set_description(f"Epoch {epoch} - Loss: {loss_num:.4f}, CE Loss: {ce_loss:.4f}, EWC Loss: {ewc_loss:.4f}")
                progress_bar.update(1)
                
                # 梯度累积和优化步骤
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)

            if use_wandb and accelerator.is_main_process:
                wandb.log({'s_loss': loss_num,
                           's_ce_loss': ce_loss})
        progress_bar.close()
        end_time = time.time()
        elapsed_time = end_time - start_time
        accelerator.print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}, Ended: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}, Duration: {elapsed_time:.2f} seconds")
        
        train_epoch_loss = total_loss / len(train_dataloader)
        accelerator.print(f"{epoch=}: {train_epoch_loss=}")

        # saving model
        accelerator.print(f"Saving model to {output_dir}...")
        accelerator.unwrap_model(model).save_pretrained(output_dir)

        if use_wandb and accelerator.is_main_process:
            wandb.log({'train_loss': train_epoch_loss})

        accelerator.wait_for_everyone()


if __name__ == "__main__":
    fire.Fire(main)

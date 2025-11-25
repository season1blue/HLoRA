'''
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
'''
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig,LlamaForCausalLM,AutoModelForCausalLM,AutoTokenizer
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

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
from utils.pile import pile_dataloader
import ipdb
import pickle
import torch.distributed as dist

from typing import Tuple



class SILoRAModel(torch.nn.Module):
    def __init__(self, model_name_or_path, fisher_matrix_path, accelerator, peft_config, pad_token, scaling):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_8bit=True, device_map={"": accelerator.local_process_index})
        self.model = prepare_model_for_int8_training(self.model)
        self.model = get_peft_model(self.model, peft_config)

        self.scaling = scaling
        self.damping_factor = 1e-9
        self.current_task_name = ''
        self.task_names = []
        self.grads = {}
        self.n_tasks = 2 #应该注意
        self.init_grads()

    def init_grads(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad: # reduce memory usage
                self.grads[n] = torch.ones([p.data.numel(), self.n_tasks], dtype=p.dtype)

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

        loss = ce_loss
        outputs.loss = loss
        outputs.ce_loss = ce_loss

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
    A_num_epochs=5,
    B_num_epochs=5,
    C_num_epochs=5,
    D_num_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    seed=42,
    max_src_len_B=500,
    max_src_len_A=800,
    max_tgt_len=256,
    ewc_lambda=1,
    num_beams=1,
    output_dir="output",
    lora_r=8,
    lora_alpha=32,
    use_wandb=False,
    A_template_name="pile",
    B_template_name="medmcqa",
    C_template_name="piqa",
    D_template_name="sciq",
    A_train_file=None,
    B_train_file=None,
    C_train_file=None,
    D_train_file=None,
    reg_conf = 1, # reg重要因子
    cuda=0
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

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    # creating model
    scaling = lora_alpha / lora_r
    
    model = SILoRAModel(model_name_or_path, fisher_matrix_path, accelerator, peft_config, tokenizer.pad_token,scaling = scaling)
    model.print_trainable_parameters()
    model.task_names.append(B_template_name)
    model.task_names.append(C_template_name)


    B_train_dataset = load_dataset("json", data_files={'train': B_train_file})['train']
    C_train_dataset = load_dataset("json", data_files={'train': C_train_file})['train']
    # D_train_dataset = load_dataset("json", data_files={'train': D_train_file})['train']

    def preprocess_function(examples, template_name):
        # ipdb.set_trace()
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
            model_input = examples['input']
        elif template_name =="pile":
            model_input = examples['text']

        if template_name =="pile":
            batch = tokenizer(
                    model_input,
                    max_length=max_src_len_A,
                    padding='max_length',
                    truncation=True,
                    add_special_tokens=False,
                    return_tensors='pt',
                )
        else:
            batch = tokenizer(
                    model_input,
                    max_length=max_src_len_B,
                    padding='max_length',
                    truncation=True,
                    add_special_tokens=False,
                    return_tensors='pt',
                )

        batch['labels'] = batch['input_ids'][:, 1:]
        batch['input_ids'] = batch['input_ids'][:, :-1]
        batch['attention_mask'] = batch['attention_mask'][:, 1:]
           
        return batch

    # STEP 2 : TRAIN B

    with accelerator.main_process_first():
        B_train_dataset = B_train_dataset.map(
            partial(preprocess_function, template_name=B_template_name),
            batched=True,
            num_proc=1,
            remove_columns=B_train_dataset.column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()

    B_data_total_size = len(B_train_dataset)
    if B_data_total_size > 20000:
        indices = np.random.choice(B_data_total_size, 20000, replace=False)
        B_train_dataset =Subset(B_train_dataset,indices)
    

    B_train_dataloader = DataLoader(
        B_train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=per_device_train_batch_size, pin_memory=True
    )

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # lr scheduler
    B_lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(B_train_dataloader) * B_num_epochs),
    )


    model, B_train_dataloader, optimizer, B_lr_scheduler = accelerator.prepare(
        model, B_train_dataloader, optimizer, B_lr_scheduler
    )

    accelerator.print(model)

    model.current_task_name = B_template_name

    print(f"curent_task:{model.current_task_name}")
    print(f"task_names:{model.task_names}")
    # Step 1: Train Task B
    for epoch in range(B_num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        total_steps = len(B_train_dataloader)
        progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process, ncols=200)

        for batch in B_train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                loss_num = loss.detach().cpu().item()
                total_loss += loss_num
                ce_loss = outputs.ce_loss.detach().cpu().item()
                # 更新 tqdm 描述信息
                progress_bar.set_description(f"TaskB: Epoch {epoch} - Loss: {loss_num:.4f}, CE Loss: {ce_loss:.4f}")
                progress_bar.update(1)

                # 梯度累积和优化步骤
                accelerator.backward(loss)
                for n, p in model.named_parameters():
                    if n in model.grads:
                        # p_grad = safe_get_full_grad(p)
                        # print('rank', dist.get_rank(), n, '->', p.device)
                        # assert p.grad is not None, f"parameter {n} has no gradient"
                        if p.grad is None:
                            print(f"rank {dist.get_rank()} parameter {n} has no gradient in device {p.device}")
                        p.grad = model.get_updated_grads(n, p.grad, model.task_names.index(model.current_task_name))
                        grad_old = model.grads[n][:, model.task_names.index(model.current_task_name)].detach().clone()
                        grad_new = (grad_old * accelerator.state.global_step + p.grad.detach().clone().view(-1)) / (accelerator.state.global_step + 1)
                        model.grads[n][:, model.task_names.index(model.current_task_name)] = grad_new

                optimizer.step()
                B_lr_scheduler.step()
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
        
        train_epoch_loss = total_loss / len(B_train_dataloader)
        accelerator.print(f"{epoch=}: {train_epoch_loss=}")

        # Saving model
        epoch_output_dir = os.path.join(output_dir, f"{B_template_name}_epoch_{epoch + 1}")
        os.makedirs(epoch_output_dir, exist_ok=True)
        accelerator.print(f"Saving model to {epoch_output_dir}...")
        accelerator.unwrap_model(model).save_pretrained(epoch_output_dir)
        
        if use_wandb and accelerator.is_main_process:
            wandb.log({'train_loss': train_epoch_loss})

        accelerator.wait_for_everyone()



    with accelerator.main_process_first():
        C_train_dataset = C_train_dataset.map(
            partial(preprocess_function, template_name=C_template_name),
            batched=True,
            num_proc=1,
            remove_columns=C_train_dataset.column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()

    C_data_total_size = len(C_train_dataset)
    if C_data_total_size > 20000:
        indices = np.random.choice(C_data_total_size, 20000, replace=False)
        C_train_dataset =Subset(C_train_dataset,indices)
    

    C_train_dataloader = DataLoader(
        C_train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=per_device_train_batch_size, pin_memory=True
    )

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # lr scheduler
    C_lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(C_train_dataloader) * C_num_epochs),
    )


    model, C_train_dataloader, optimizer, C_lr_scheduler = accelerator.prepare(
        model, C_train_dataloader, optimizer, C_lr_scheduler
    )

    accelerator.print(model)

    model.current_task_name = C_template_name
    print(f"curent_task:{model.current_task_name}")
    print(f"task_names:{model.task_names}")


    # Step 1: Train Task C
    for epoch in range(C_num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        total_steps = len(C_train_dataloader)
        progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process, ncols=200)

        for batch in C_train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                loss_num = loss.detach().cpu().item()
                total_loss += loss_num
                ce_loss = outputs.ce_loss.detach().cpu().item()
                # 更新 tqdm 描述信息
                progress_bar.set_description(f"TaskC: Epoch {epoch} - Loss: {loss_num:.4f}, CE Loss: {ce_loss:.4f}")
                progress_bar.update(1)

                # 梯度累积和优化步骤
                accelerator.backward(loss)

                for n, p in model.named_parameters():
                    if n in model.grads:
                        # p_grad = safe_get_full_grad(p)
                        # print('rank', dist.get_rank(), n, '->', p.device)
                        # assert p.grad is not None, f"parameter {n} has no gradient"
                        if p.grad is None:
                            print(f"rank {dist.get_rank()} parameter {n} has no gradient in device {p.device}")
                        p.grad = model.get_updated_grads(n, p.grad, model.task_names.index(model.current_task_name))
                        grad_old = model.grads[n][:, model.task_names.index(model.current_task_name)].detach().clone()
                        grad_new = (grad_old * accelerator.state.global_step + p.grad.detach().clone().view(-1)) / (accelerator.state.global_step + 1)
                        model.grads[n][:, model.task_names.index(model.current_task_name)] = grad_new
                        
                optimizer.step()
                C_lr_scheduler.step()
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
        
        train_epoch_loss = total_loss / len(C_train_dataloader)
        accelerator.print(f"{epoch=}: {train_epoch_loss=}")

        # Saving model
        epoch_output_dir = os.path.join(output_dir, f"{C_template_name}_epoch_{epoch + 1}")
        os.makedirs(epoch_output_dir, exist_ok=True)
        accelerator.print(f"Saving model to {epoch_output_dir}...")
        accelerator.unwrap_model(model).save_pretrained(epoch_output_dir)
        
        if use_wandb and accelerator.is_main_process:
            wandb.log({'train_loss': train_epoch_loss})

        accelerator.wait_for_everyone()
    

if __name__ == "__main__":
    fire.Fire(main)
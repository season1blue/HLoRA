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
from utils.pile import pile_dataloader
import ipdb
import pickle

from typing import Tuple

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1

def concat_inputs(input_ids:torch.Tensor, attention_mask:torch.Tensor, labels:torch.Tensor, buffer_input_ids:torch.Tensor, buffer_attention_mask:torch.Tensor, buffer_labels:torch.Tensor) -> Tuple:
    device = input_ids.device
    input_ids = torch.cat((input_ids, buffer_input_ids.to(device)), dim=0)
    attention_mask = torch.cat((attention_mask, buffer_attention_mask.to(device)), dim=0)
    labels = torch.cat((labels, buffer_labels.to(device)), dim=0)
    return input_ids, attention_mask, labels

class Buffer:
    def __init__(self, buffer_size:int, device:str, pad_id:int=2, ignore_index:int=-100):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ['input_ids', 'attention_mask', 'labels', 'logits', 'task_labels', 'activations']
        self.init_buffer()
        self.pad_id = pad_id
        self.ignore_index = ignore_index
        
    def init_buffer(self) -> None:
        for attr_str in self.attributes:
            setattr(self, attr_str, [None for _ in range(self.buffer_size)])

    def add_data(self, input_ids, attention_mask=None, labels=None, logits=None, task_labels=None, activations=None):
        n = input_ids.shape[0] if hasattr(input_ids, 'shape') else len(input_ids)
        for i in range(n):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.input_ids[index] = input_ids[i].detach().clone().to(self.device)
                if attention_mask is not None:
                   self.attention_mask[index] = attention_mask[i].detach().clone().to(self.device) 
                if labels is not None:
                    self.labels[index] = labels[i].detach().clone().to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].detach().clone().to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].detach().clone().to(self.device)
                if activations is not None:
                    self.activations[index] = activations[i].detach().clone().to(self.device)

    def get_data(self, size: int, pad_to:int) -> Tuple:
        n = len(self.input_ids)
        if size > min(self.num_seen_examples, n):
            size = min(self.num_seen_examples, n)

        choice = np.random.choice(min(self.num_seen_examples, n), size=size, replace=False)
        if len(choice) == 0:
            return None, None
        # for left padding
        input_ids = []
        attention_mask = []
        labels = []
        
        for i in choice:

            input_ids.append(torch.cat(
                (torch.full((pad_to - self.input_ids[i].shape[-1],), self.pad_id, dtype=torch.long).to(self.device),
                self.input_ids[i]), dim=-1)
            )
            if self.attention_mask[i] is not None:
                attention_mask.append(torch.cat(
                    (torch.full((pad_to - self.attention_mask[i].shape[-1],), 0, dtype=torch.long).to(self.device),
                    self.attention_mask[i]), dim=-1)
                )
            if self.labels[i] is not None:
                labels.append(torch.cat(
                    (torch.full((pad_to - self.labels[i].shape[-1],), self.ignore_index, dtype=torch.long).to(self.device),
                    self.labels[i]), dim=-1)
                )
        
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        labels = torch.stack(labels)
        return input_ids, attention_mask, labels

    def is_empty(self) -> bool:
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self) -> Tuple:
        ret_tuple = (torch.stack([ee.cpu()
                                  for ee in self.input_ids]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0

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
        self.buffer_size = 100
        self.buffer = Buffer(self.buffer_size, 'cpu', pad_id=pad_token, ignore_index=-100)

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

        
    def forward(self, **kwargs):
        labels = kwargs["labels"]
        # label_weights = kwargs.pop("label_weights")

        if self.current_task_name == self.task_names[0]:
            self.buffer.add_data(kwargs["input_ids"], kwargs["attention_mask"], kwargs["labels"])
            outputs = self.model(**kwargs)
        else:
            buffer_inputs, buffer_attention_mask, buffer_labels = self.buffer.get_data(kwargs["input_ids"].shape[0], kwargs["input_ids"].shape[1])
            if buffer_inputs is not None and buffer_attention_mask is not None and buffer_labels is not None:
                kwargs["input_ids"], kwargs["attention_mask"], kwargs["labels"] = concat_inputs(kwargs["input_ids"], kwargs["attention_mask"], kwargs["labels"], buffer_inputs, buffer_attention_mask, buffer_labels) 
            outputs = self.model(**kwargs)
            self.buffer.add_data(kwargs["input_ids"], kwargs["attention_mask"], kwargs["labels"])
        
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
    
    model = SILoRAModel(model_name_or_path, fisher_matrix_path, accelerator, peft_config, tokenizer.pad_token_id,scaling = scaling)
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
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

import torch.nn as nn
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

class SILoRAModel(torch.nn.Module):
    def __init__(self, model_name_or_path, fisher_matrix_path, accelerator, peft_config, scaling):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_8bit=True, device_map={"": accelerator.local_process_index})
        self.model = prepare_model_for_int8_training(self.model)
        self.model = get_peft_model(self.model, peft_config)

        self.scaling = scaling
        self.damping_factor = 1e-9
        self.params = self.get_lora_params()
        self.w = {}  # Parameters contribution to change in loss

        for n, p in self.params.items():
            self.w[n] = p.zero_()

        self.initial_params = {}
        for n, p in self.params.items():
            self.initial_params[n] = p
            
        self.regularization_terms = {}
        self.lora_scaling_factors = nn.ParameterDict()
        for name in self.get_lora_params().keys():
            # print(type(name))
            name = name.replace('.', '*')
            # print(name)
            self.lora_scaling_factors[name] = nn.Parameter(torch.tensor(1.0))

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def get_lora_params(self, need_detach=True):

        params = {}

        for name, module in self.model.named_modules():
            if isinstance(module, LoraLayer):
                if module.active_adapter not in module.lora_A.keys():
                    continue
                if isinstance(module, bnb.nn.Linear8bitLt):
                    fan_in_fan_out=False
                else:
                    fan_in_fan_out = module.fan_in_fan_out
                if need_detach == True:
                    adapter_weights = transpose(
                        module.lora_B[module.active_adapter].weight.detach() @ module.lora_A[module.active_adapter].weight.detach(),
                        fan_in_fan_out,
                    ) * module.scaling[module.active_adapter]
                else:
                    adapter_weights = transpose(
                        module.lora_B[module.active_adapter].weight @ module.lora_A[module.active_adapter].weight,
                        fan_in_fan_out,
                    ) * module.scaling[module.active_adapter]

                params[name] = adapter_weights
        
        return params

    def save_imporatnce(self, importance_name):
        if "A" in self.regularization_terms.keys():
            importance_dict = self.regularization_terms["A"]['importance']
        else:
            importance_dict = None
            
        with open(f'{importance_name}.pkl', 'wb') as file:
            pickle.dump(importance_dict, file)
        
    def calculate_importance(self, task="None"):  #涉及None阶段初始化Importance，以及A阶段计算A任务的Importance
            """
            Calculate the importance of each parameter for SI.
            """

            print(f"Cal Importance of {task}")
            # Initialize the importance matrix
            if task=="None": # The case of the first task
                importance = {}
                for n, p in self.params.items():
                    importance[n] = p.fill_(0)
                self.regularization_terms["None"] = {'importance': importance, 'task_param': self.initial_params}
                # ipdb.set_trace()
                
            elif task=="A":    # The case of after the first task
                importance = self.regularization_terms["None"]['importance']
                prev_params = self.regularization_terms["None"]['task_param']

                # 更新Importance和prev_params
                cur_params = self.get_lora_params()

                for name, par in importance.items():
                    delta_theta = cur_params[name]
                    importance[name] = importance[name] + (self.w[name] / (delta_theta ** 2 + self.damping_factor))
                    # print(f"importance of {name}: {importance[name]}")
                    self.w[name].zero_()

                # 以当前的model_params作为代表A阶段的prev_params
                prev_params = {}
                for n, p in cur_params.items():
                    prev_params[n] = p
                
                self.regularization_terms["A"] = {'importance': importance, 'task_param': prev_params}  # 留个下个任务用

            return importance
    def save_importance(self, importance_name):
            if "A" in self.regularization_terms.keys():
                importance_dict = self.regularization_terms["A"]['importance']
            else:
                importance_dict = None
                
            with open(f'{importance_name}.pkl', 'wb') as file:
                pickle.dump(importance_dict, file)

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
    num_epochs=5,
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
    A_train_file=None,
    B_train_file=None,
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
    model = SILoRAModel(model_name_or_path, fisher_matrix_path, accelerator, peft_config, scaling = scaling)
    model.print_trainable_parameters()


    A_train_dataset = load_dataset("json", data_files={'train': A_train_file})['train']
    B_train_dataset = load_dataset("json", data_files={'train': B_train_file})['train']


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

    with accelerator.main_process_first():
        A_train_dataset = A_train_dataset.map(
            partial(preprocess_function, template_name=A_template_name),
            batched=True,
            num_proc=1,
            remove_columns=A_train_dataset.column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()

    A_data_total_size = len(A_train_dataset)
    if A_data_total_size > 2000:
        indices = np.random.choice(A_data_total_size, 2000, replace=False)
        A_train_dataset =Subset(A_train_dataset,indices)
    

    A_train_dataloader = DataLoader(
        A_train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=per_device_train_batch_size, pin_memory=True
    )

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # lr scheduler
    A_lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(A_train_dataloader) * num_epochs),
    )


    model, A_train_dataloader, optimizer, A_lr_scheduler = accelerator.prepare(
        model, A_train_dataloader, optimizer, A_lr_scheduler
    )

    accelerator.print(model)

    model.calculate_importance(task="None") ## 初始化重要性矩阵和记录prev_params

    # Step 1: Train Task A
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        total_steps = len(A_train_dataloader)
        progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process, ncols=200)

        for batch in A_train_dataloader:
            with accelerator.accumulate(model):
                unreg_gradients = {}

                # 在每个batch开始获取参数，赋值给old_params
                cur_params = model.get_lora_params()
                old_params = {}
                for n, p in cur_params.items():
                    old_params[n] = p

                # 更新参数的过程
                outputs = model(**batch)
                loss = outputs.loss
                loss_num = loss.detach().cpu().item()
                total_loss += loss_num
                ce_loss = outputs.ce_loss.detach().cpu().item()
                # 更新 tqdm 描述信息
                progress_bar.set_description(f"TaskA: Epoch {epoch} - Loss: {loss_num:.4f}, CE Loss: {ce_loss:.4f}")
                progress_bar.update(1)

                # 梯度累积和优化步骤
                accelerator.backward(loss)

                 # 计算每个batch中的梯度
                for name, module in model.named_modules():
                    if isinstance(module, LoraLayer):
                        if module.active_adapter not in module.lora_A.keys():
                            continue
                        
                        lora_a  = module.lora_A[module.active_adapter].weight.detach()
                        lora_b  = module.lora_B[module.active_adapter].weight.detach()
                        lora_a_grad = module.lora_A[module.active_adapter].weight.grad.detach()
                        lora_b_grad = module.lora_B[module.active_adapter].weight.grad.detach()
                    

                        # if torch.count_nonzero(lora_b) != 0:
                        #     a = lora_b @ transpose(lora_b,fan_in_fan_out=True)
                        #     b = a @ lora_b @ lora_a_grad
                        #     print(f"gradient1:\n{b}")

                        #     c = transpose(lora_a,fan_in_fan_out=True) @ lora_a
                        #     d=  lora_b_grad @ lora_a @ c
                        #     print(f"gradient2:\n{d}")

                        # print(f"lora_a_grad.size:{lora_a_grad.size()}")
                        #计算gradients
                        name = name.replace('model.base_model', 'base_model')
                        current_lr = optimizer.param_groups[0]['lr']
                        unreg_gradients[name] = module.scaling[module.active_adapter] * (lora_b @ lora_a_grad + lora_b_grad @ lora_a - current_lr * lora_b_grad @ lora_a_grad)
                        # print(f"unreg_gradients[name]:{name}\n")

                optimizer.step()
                A_lr_scheduler.step()

                # 获取更新之后的参数
                cur_params = model.get_lora_params()

                for name, param in cur_params.items():
                    delta = param - old_params[name]
                    # delta = param
                    if name in unreg_gradients.keys():
                        model.w[name] -= unreg_gradients[name] * delta  # w[n] is >=0
                        # model.w[name] += (delta ** 2) / current_lr  # w[n] is >=0
                        model.w[name][model.w[name] < 0 ] = 0
                        # print(f"unreg_gradients of {name} :{unreg_gradients[name]}\n")
                        # print(f"delta of {name} :{delta}\n")
                        # print(f"gradient*delta of {name} :{unreg_gradients[name] * delta}\n")
                # a = model.w['base_model.model.transformer.h.21.attn.q_proj.lora.default.weight']
                # print(f"model.w:{a}")

                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)

            if use_wandb and accelerator.is_main_process:
                wandb.log({'s_loss': loss_num,
                           's_ce_loss': ce_loss})
        progress_bar.close()

   # Step 2: Calculate the importance of  Task A
        model.calculate_importance(task="A")
        model.save_imporatnce(
            importance_name=f"importance_{B_template_name}_llama_{reg_conf}"
            )  #e.g. B_template_name="medmcqa", reg_conf=1e-3

        # model.save_importance(
        #     importance_name=f"importance_{B_template_name}_gptj_{reg_conf}"
        #     )  #e.g. B_template_name="medmcqa", reg_conf=1e-3

        end_time = time.time()
        elapsed_time = end_time - start_time
        accelerator.print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}, Ended: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}, Duration: {elapsed_time:.2f} seconds")
        
        train_epoch_loss = total_loss / len(A_train_dataloader)
        accelerator.print(f"{epoch=}: {train_epoch_loss=}")

        if use_wandb and accelerator.is_main_process:
            wandb.log({'train_loss': train_epoch_loss})

        accelerator.wait_for_everyone()

    # Step 3: Train Task B

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

    B_lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(B_train_dataloader) * num_epochs),
    )

    model, B_train_dataloader, optimizer, B_lr_scheduler = accelerator.prepare(
        model, B_train_dataloader, optimizer, B_lr_scheduler
    )

    for epoch in range(num_epochs):

            model.train()
            total_loss = 0

            start_time = time.time()
            total_steps = len(B_train_dataloader)
            progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process, ncols=200)

            for batch in B_train_dataloader:
                with accelerator.accumulate(model):
                    reg_loss = 0
                    task_reg_loss = 0
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss_num = loss.detach().cpu().item()
                    total_loss += loss_num
                    ce_loss = outputs.ce_loss.detach().cpu().item()

                    importance = model.regularization_terms["A"]['importance']
                    task_param = model.regularization_terms["A"]['task_param']

                    dynamic_params = model.get_lora_params(need_detach=False)

                    l2_norms = {}
                    for n, p in dynamic_params.items():
                        l2_norms[n] = np.linalg.norm(importance[n].cpu().numpy())
                    l2_values = np.array(list(l2_norms.values()))
                    l2_values_exp = np.exp(l2_values*1e-8)  # 计算 e^x
                    softmax_values = l2_values_exp / np.sum(l2_values_exp)  # Softmax 归一化
                    softmax_norms = {name: softmax_values[i] for i, name in enumerate(dynamic_params.keys())}

                    for index, (n, p) in enumerate(dynamic_params.items()):
                        curr_norm = softmax_norms[n]
                        n_scaling = n.replace('.', '*')
                        scaling_factor = model.lora_scaling_factors[n_scaling]
                        task_reg_loss += scaling_factor * curr_norm * (importance[n] * (p - task_param[n]) ** 2).sum()

                    reg_loss += task_reg_loss
                    loss += reg_loss * reg_conf
                    # ipdb.set_trace()

                    # 更新 tqdm 描述信息
                    progress_bar.set_description(f"TaskB: Epoch {epoch} - Loss: {loss:.4f},Ce_Loss: {loss_num:.4f} Reg_Loss: {reg_loss:.4f}")
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

            # saving model

            epoch_output_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
            os.makedirs(epoch_output_dir, exist_ok=True)
            accelerator.print(f"Saving model to {epoch_output_dir}...")
            accelerator.unwrap_model(model).save_pretrained(epoch_output_dir)

            if use_wandb and accelerator.is_main_process:
                wandb.log({'train_loss': train_epoch_loss})

            accelerator.wait_for_everyone()

if __name__ == "__main__":
    fire.Fire(main)
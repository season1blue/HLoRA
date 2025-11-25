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
from torch import nn
from copy import deepcopy

def l2_normalize(x, dim=None, epsilon=1e-12):
    square_norm = torch.sum(x ** 2, dim=dim, keepdim=True)
    x_inv_norm = torch.rsqrt(torch.maximum(square_norm, torch.tensor(epsilon, device=x.device)))
    return x * x_inv_norm



class EWCLoRAModel(torch.nn.Module):
    def __init__(self, model_name_or_path, fisher_matrix_path, accelerator, ewc_lambda=1):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_8bit=True, device_map={"": accelerator.local_process_index})
        self.model = prepare_model_for_int8_training(self.model)
        self.embed_tokens = self.model.get_input_embeddings()
        self.embed_tk_shapes = self.embed_tokens.weight.shape
        self.prompt = None
        self.top_k = 3
        self.diversity_loss_weight = 0.5
        self.pool_size = 10
        self.prompt_length = 5
        self.device ='cpu'

        self.init_prompt('random')
        self.embeding_key = 'mean'
        self.batchwise_prompt: bool = False
        self.current_task_name:str = None
        self.ewc_lambda = ewc_lambda

        
    def init_prompt(self,promt_init):
        self.prompt = nn.Parameter(
            torch.tensor(
                self.create_prompt(self.pool_size, self.prompt_length, promt_init), requires_grad=True
            )
        ).to(self.device)

    def create_prompt(self, pool_size, prompt_length, promt_init='random'):
        N = self.embed_tk_shapes[0]
        p_weights = []
        
        for p in range(self.pool_size):
            p_w = []
            for i in range(self.prompt_length):
                with torch.no_grad():
                    j = np.random.randint(N)
                    w = deepcopy(self.embed_tokens.weight[j].detach().cpu().numpy())
                    p_w.append(w)
            p_weights.append(p_w)
            
        return np.array(p_weights)

    def save_prompt_weights(self, path):
        state_dict = {"prompt_pool": self.prompt}
        torch.save(state_dict, os.path.join(path, f"prompt_weights_{self.current_task_name}.pt"))
    
    def load_prompt_weights(self, path, task_name="jecqa"):
        state_dict = torch.load(os.path.join(path, f"prompt_weights_{task_name}.pt"), map_location=self.device)
        self.prompt.data = state_dict["prompt_pool"].data
        print(f"Loaded prompt weights from {path}")
        
    def freeze_prompt(self):
        for n, p in self.named_parameters():
            p.requires_grad = False

        # self.fisher_matrix = AutoModelForCausalLM.from_pretrained(fisher_matrix_path, load_in_8bit=True, device_map={"": accelerator.local_process_index})
        # self.fisher_matrix.eval()
        # self.fisher_matrix.requires_grad_(False)


    def get_peft_model(self, peft_config):
        self.model = get_peft_model(self.model, peft_config)

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]
        # inputs_embeds = kwargs["inputs_embeds"]

        labels = kwargs.pop("labels")


        i_input_embeds = self.embed_tokens(input_ids)
        out = dict()
        if self.embeding_key == 'mean':
            i_input_embeds_mean = torch.mean(i_input_embeds, dim=1)
        elif self.embeding_key == 'max':
            i_input_embeds_mean = torch.max(i_input_embeds, dim=1)[0]
        elif self.embeding_key == 'mean_max':
            i_input_embeds_mean = torch.max(i_input_embeds, dim=1)[0] + 2 * torch.mean(i_input_embeds, dim=1)
        else:
            raise NotImplementedError("Not supported way of calculating embedding keys!")
        
        prompt_key = torch.mean(self.prompt, dim=1) # Pool_size, C
        prompt_norm = l2_normalize(prompt_key, dim=1).to("cuda")
        inputs_embeds_norm = l2_normalize(i_input_embeds_mean, dim=1)
        prompt_norm = prompt_norm.to(dtype=inputs_embeds_norm.dtype)
        similarity = torch.matmul(inputs_embeds_norm, prompt_norm.t()) # B, Pool_size
        
        _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
        if self.batchwise_prompt:
            prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
            if prompt_id.shape[0] < self.pool_size:
                prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
            _, major_idx = torch.topk(id_counts, k=self.top_k)
            major_prompt_id = prompt_id[major_idx]
            idx = major_prompt_id.expand(inputs_embeds.shape[0], -1)
        
        batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
        batch_size, top_k, length, c = batched_prompt_raw.shape
        batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)
        inputs_embeds = torch.cat([batched_prompt, i_input_embeds],axis=1)
        
        prefix_length = batched_prompt.shape[1]
        attn_masks = torch.concat((torch.tensor(1).to("cuda").repeat(batch_size,prefix_length),attention_mask), axis=1)
        
        if labels is None: # inference
            return self.model(inputs_embeds=inputs_embeds.half(), attention_mask=attn_masks, use_cache=False, return_dict=True)
        
        labels = torch.concat((labels[0][0].repeat(batch_size,inputs_embeds.shape[1]-labels.shape[1]),labels),axis=1)
        outs = self.model(inputs_embeds=inputs_embeds, attention_mask=attn_masks)
        logits = outs.logits
        ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        ce_loss = ce_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        ce_loss = torch.mean(ce_loss)
        loss = ce_loss

        batched_key_norm = prompt_norm[idx]
        inputs_embeds_norm = inputs_embeds_norm.unsqueeze(1) # B, 1, C
        sim = batched_key_norm * inputs_embeds_norm # B, top_k, C
        reduce_sim = torch.sum(sim) / inputs_embeds.shape[0] # Scalar

        loss = torch.abs(loss - reduce_sim * self.diversity_loss_weight)

        outs.loss = loss
        outs.ce_loss = ce_loss
        return outs

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
                # ewc_loss = outputs.ewc_loss.detach().cpu().item()
                
                # 更新 tqdm 描述信息
                progress_bar.set_description(f"Epoch {epoch} - Loss: {loss_num:.4f}, CE Loss: {ce_loss:.4f}")
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
        model.save_prompt_weights(output_dir)

        if use_wandb and accelerator.is_main_process:
            wandb.log({'train_loss': train_epoch_loss})

        accelerator.wait_for_everyone()


if __name__ == "__main__":
    fire.Fire(main)

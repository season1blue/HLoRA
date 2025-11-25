import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)
from typing import Dict
import time
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from peft.utils import transpose
import fire
from functools import partial
import bitsandbytes as bnb
from peft import prepare_model_for_int8_training, PeftModel
import math
import wandb
import math
from copy import deepcopy
import torch.nn as nn
from sciq import sciq_dataloader
from pile import pile_dataloader
from piqa import piqa_dataloader
from medmcqa import medmcqa_dataloader
import os
from regularization import SI
from data_loader import domain_dataloader
import ipdb
import logging

from transformers import Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # 输出到控制台
)





def main(
    model_name_or_path="EleutherAI/gpt-neo-1.3B",
    lora_path="output/lora-gptj-sciq/checkpoint",
    A_train_file=None,
    B_train_file=None,
    text_column="input",
    label_column="ref",
    lr=1e-3,
    num_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    label_file=None,
    seed=42,
    max_src_len=800,
    max_tgt_len=256,
    ewc_lambda=1,
    num_beams=1,
    output_dir="output",
    lora_r=8,
    comments="None",
    lora_alpha=32,
    use_wandb=False,
    template_name="medmcqa"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    kwargs = DistributedDataParallelKwargs(static_graph=True)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=lora_r, 
        lora_alpha=lora_alpha, 
        lora_dropout=0.1
    )
    set_seed(seed)

    # 创建任务 A 和任务 B 的数据加载器
    A_dataloader = pile_dataloader(A_train_file, model_name_or_path, max_src_len, per_device_train_batch_size)
    B_dataloader = domain_dataloader(B_train_file, template_name, model_name_or_path, max_src_len, per_device_train_batch_size)


    base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_8bit=True)
    base_model = get_peft_model(base_model, peft_config)
    # 创建模型
    scaling = lora_alpha / lora_r
    model = SI(base_model, lr, scaling)
    model = model.to(device)  # 将模型转移到 CUDA 或 CPU
    # model.get_peft_model(peft_config)


    # Step 1: Train Task A
    logging.info("Cal Importance of None")
    model.calculate_importance(task="None") ## 初始化重要性矩阵和记录prev_params
    A_num_epochs = 5
    for epoch in range(A_num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        total_steps = len(A_dataloader)
        progress_bar = tqdm(enumerate(A_dataloader), total=total_steps, ncols=100)
        # ipdb.set_trace()
        for i,batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model.update_model(**batch, task="A")
            total_loss += loss.item()
            
            # if i % 10 == 0:
            progress_bar.set_description(f"Task A, E{epoch}-It{i}, Loss: {loss.item():.4f}")
            # progress_bar.update(10)  # 每10步更新进度条
            # progress_bar.set_postfix(loss=total_loss)

        avg_loss = total_loss / len(A_dataloader)
        logging.info(f"AVG LOSS: {avg_loss}")
        # ipdb.set_trace()
        logging.info("Cal Importance of A")
        model.calculate_importance(task="A") ## 计算A任务的重要性矩阵

        progress_bar.close()
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Task A: {elapsed_time:.2f} seconds, Train Loss: {avg_loss:.4f}")

    # if total_loss != total_loss:  # 判断是否为 NaN
    #     print(f"Epoch {epoch + 1} Task A - NaN detected in loss, stopping training...")
    #     task_b_output_dir = os.path.join(output_dir, "task_b_model")
    #     print(f"Saving Task B model to {task_b_output_dir}...")
    #     model.save_pretrained(task_b_output_dir)


    for epoch in range(num_epochs):
        # Step 2: Train Task B
        model.train()
        total_loss = 0
        start_time = time.time()
        total_steps = len(B_dataloader)
        progress_bar = tqdm(enumerate(B_dataloader), total=total_steps, ncols=100)

        for i, batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model.update_model(**batch, task="B")

            if torch.isnan(loss):  # 判断是否为 NaN
                logging.info(f"Epoch {epoch + 1} Task B - NaN detected in loss, stopping training...")
                task_b_output_dir = os.path.join(output_dir, "task_b_model_earlystop")
                model.save_pretrained(task_b_output_dir)
                logging.info(f"Saving Task B model to {task_b_output_dir}...")
                exit()
            
            total_loss += loss.item()
            # 更新 tqdm 描述信息
            progress_bar.set_description(f"Task B, E{epoch}-It{i}, Loss: {loss.item():.4f}")
            # progress_bar.update(1)
        avg_loss = total_loss / len(B_dataloader)
        progress_bar.close()
        end_time = time.time()
        elapsed_time = end_time - start_time


        logging.info(f"Task B: {elapsed_time:.2f} seconds, Train Loss: {avg_loss:.4f}")

        task_b_output_dir = os.path.join(output_dir, comments)
        logging.info(f"Saving Task B model to {task_b_output_dir}...")
        model.save_pretrained(task_b_output_dir)



if __name__ == "__main__":
    fire.Fire(main)
    # main()

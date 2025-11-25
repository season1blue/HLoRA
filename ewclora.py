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
import math
from utils.data_loader import domain_dataloader

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
        outputs = self.model(**kwargs)

        # EWC loss
        fisher_matrix_module_dict = {name: module for name, module in self.fisher_matrix.named_modules()}
        ewc_loss = 0
        for name, module in self.model.named_modules():

            if isinstance(module, LoraLayer):
                print(f"module.active_adapter:{module.active_adapter}")
                print(f"module.lora_A.keys():{module.lora_A.keys()}")
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

        outputs.ewc_loss = ewc_loss
        outputs.ce_loss = outputs.loss
        outputs.loss += self.ewc_lambda * ewc_loss
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
    label_file="datasets/piqa/train-labels.lst",
    text_column="input",
    label_column="ref",
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
    if use_wandb:
        wandb_args = {'model': model_name_or_path.split('/')[-1],
                    'train_file': train_file,
                    'lr': lr,
                    'num_epochs': num_epochs,
                    'per_device_train_batch_size': per_device_train_batch_size,
                    'ewc_lambda': ewc_lambda,
                    'lora_r': lora_r,
                    'lora_alpha': lora_alpha,
                    'num_beams': num_beams}
        
    kwargs = DistributedDataParallelKwargs(static_graph=True)
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, kwargs_handlers=[kwargs])
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=lora_r, 
        lora_alpha=lora_alpha, 
        lora_dropout=0.1
    )
    set_seed(seed)

    
    if use_wandb and accelerator.is_main_process:
        wandb.init(project='ewc-lora', config=wandb_args, save_code=True)


    #print(next(iter(train_dataloader)))

    # creating model
    model = EWCLoRAModel(model_name_or_path, fisher_matrix_path, accelerator, ewc_lambda=ewc_lambda)
    model.get_peft_model(peft_config)
    model.print_trainable_parameters()
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


    train_dataloader = domain_dataloader(train_file, template_name, model_name_or_path, max_src_len, per_device_train_batch_size)
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
                        #    's_ewc_loss': ewc_loss})
        progress_bar.close()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}, Ended: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}, Duration: {elapsed_time:.2f} seconds")
        
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
    # main()

import torch
import random
# from .default import NormalNN
from typing import Dict
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model
import torch.nn as nn
import torch.optim as optim
import ipdb
import os
import json


class SI(nn.Module):
    """
    Synaptic Intelligence.
    """
        
    def __init__(self, base_model, lr, scaling):
        """
        Initialize the SI agent.

        Parameters
        ----------
        agent_config : dict
            A dictionary of configuration for the agent.
            Keys:
                - epochs: int, the number of epochs.
                - lr: float, the learning rate.
                - weight_decay: float, the weight decay.
                - reg_coef: float, the regularization coefficient.
                - model_type: str, the type of the model.
                - model_name: str, the name of the model.
                - out_dim: dict, {task:dim}.
                - model_weights: str, the path to the model weights.
                - print_freq: int, the frequency of printing.
                - gpu: bool, whether to use gpu.

        Returns
        -------
        None.
        """
        super(SI, self).__init__()
        self.online_reg = True  # True: There will be only one importance matrix and previous model parameters
                                # False: Each task has its own importance matrix and model parameters
        self.damping_factor = 1e-3

        self.w = {}  # Parameters contribution to change in loss
        
        # self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_8bit=True)
        # self.model = get_peft_model(self.model, peft_config)
        self.model = base_model

        self.scaling = scaling

        self.params = self.get_lora_params()


        for n, p in self.params.items():
            self.w[n] = p.zero_()

        # The initial_params will only be used in the first task (when the regularization_terms is empty)
        self.initial_params = {}
        for n, p in self.params.items():
            self.initial_params[n] = p
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.regularization_terms = {}
        
        # for n, p in self.model.named_parameters():
        #     print(f"{n}: requires_grad={p.requires_grad}")

        # exit()
    def get_lora_params(self):

        params = {n : p for n, p in self.model.named_parameters() if p.requires_grad}
        new_params = {}
            
        # 遍历模型的每一层
        for name, param in params.items():
            if "lora_A" in name:
                # 找到对应的lora_b参数
                lora_b_name = name.replace("lora_A", "lora_B")
                if lora_b_name in params:
                    lora_a = param.detach()
                    lora_b = params[lora_b_name].detach()

                    # 计算乘积
                    lora_product = lora_b @ lora_a * self.scaling
                    
                    # 创建新的参数名称
                    new_param_name = name.replace("lora_A", "lora")
                    
                    # 将乘积存储在新的参数字典中
                    new_params[new_param_name] = lora_product

        params = new_params
        return params

    def get_peft_model(self, peft_config):
        self.model = get_peft_model(self.model, peft_config)
        
    def save_pretrained(self, *args, **kwargs):
        self.model.save_pretrained(*args, **kwargs)
        
    
    def calculate_importance(self, task="None"):  #涉及None阶段初始化Importance，以及A阶段计算A任务的Importance
        """
        Calculate the importance of each parameter for SI.
        """
        assert self.online_reg,'SI needs online_reg=True'

        print(f"Cal Importance of {task}")
        # Initialize the importance matrix
        if task=="None": # The case of the first task
            importance = {}
            for n, p in self.params.items():
                importance[n] = p.fill_(0)
            self.regularization_terms["None"] = {'importance': importance, 'task_param': self.initial_params}
            
        elif task=="A":    # The case of after the first task
            importance = self.regularization_terms["None"]['importance']
            prev_params = self.regularization_terms["None"]['task_param']

            # 更新Importance和prev_params
            self.params = self.get_lora_params()
            # ipdb.set_trace()
            for name, par in importance.items():
                # new_name = "base_model.model." + name
                delta_theta = self.params[name] - prev_params[name]
                importance[name] += self.w[name] / (delta_theta ** 2 + self.damping_factor)
                self.w[name].zero_()
            
            # print(importance)
            
            # 以当前的model_params作为代表A阶段的prev_params
            prev_params = {}
            for n, p in self.params.items():
                prev_params[n] = p
            
            self.regularization_terms["A"] = {'importance': importance, 'task_param': prev_params}  # 留个下个任务用

        return importance
        
    def update_model(self, **kwargs):  #涉及A阶段根据迭代的模型参数计算w，以及B阶段计算reg_loss
        """
        Update the model.
        """
        task = kwargs.pop("task", "A")  # 默认是A任务
        unreg_gradients = {}
        self.params = self.get_lora_params()
        # 1. Save current parameters
        old_params = {}
        for n, p in self.params.items():
            old_params[n] = p
    
        # 2. Compute the gradients of the loss w.r.t. the parameters without regularization
        labels = kwargs.pop("labels")
        # label_weights = kwargs.pop("label_weights")
        outputs = self.model(**kwargs)
        logits = outputs.logits
        ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        ce_loss = ce_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = torch.mean(ce_loss)
        
        if task == "A":
            self.optimizer.zero_grad()
            loss.backward()

            # 计算A任务的梯度
            updated_ori_params = {n : p for n, p in self.model.named_parameters() if p.requires_grad}
            for name, param in updated_ori_params.items():
                if "lora_A" in name:
                    # 找到对应的lora_b参数
                    lora_b_name = name.replace("lora_A", "lora_B")
                    if lora_b_name in updated_ori_params:
                        lora_a = param.detach()
                        lora_a_grad = param.grad.detach()
                        lora_b = updated_ori_params[lora_b_name].detach()
                        lora_b_grad = updated_ori_params[lora_b_name].grad.detach()
                        new_param_name = name.replace("lora_A", "lora")

                        # 计算gradients
                        unreg_gradients[new_param_name] = self.scaling * (lora_b @ lora_a_grad + lora_b_grad @ lora_a)
            
            self.optimizer.step()
            self.params = self.get_lora_params()
            
            # 计算A任务的w
            for name, param in self.params.items():
                delta = param - old_params[name]
                
                if name in unreg_gradients.keys():  # In multi-head network, some head could have no grad (lazy) since no loss go through it.
                    self.w[name] -= unreg_gradients[name] * delta  # w[n] is >=0

        # with reg term     regularization=True
        reg_coef = 0 # reg重要因子
        if task == "B":
            # Calculate the reg_loss only when the regularization_terms exists
            reg_loss = 0
            # Sum the regularization loss over previous tasks
            task_reg_loss = 0
            importance = self.regularization_terms["A"]['importance']
            task_param = self.regularization_terms["A"]['task_param']

            self.params = self.get_lora_params()

            for n, p in self.params.items():
                task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
            # ipdb.set_trace()
            reg_loss += task_reg_loss
            loss += reg_loss * reg_coef
            
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        

        return loss.detach()

    
    # def save_pretrained(self, save_directory: str):
    #     """
    #     Save the model's state dict and configuration to a directory.
    #     This will allow the model to be reloaded using `from_pretrained`.
    #     """
    #     os.makedirs(save_directory, exist_ok=True)

    #     # Save the model state dict
    #     model_state_dict = self.model.state_dict()
    #     torch.save(model_state_dict, os.path.join(save_directory, "pytorch_model.bin"))

    #     # Optionally, save the model's configuration (you may need to define a config class)
    #     # Here we just save the model's config dictionary as an example
    #     if hasattr(self.model.config, "to_dict"):
    #         config_dict = self.model.config.to_dict()
    #     else:
    #         config_dict = self.config  # If you have custom configuration for SI model
    #     with open(os.path.join(save_directory, "config.json"), "w") as f:
    #         json.dump(config_dict, f)

    #     print(f"Model saved to {save_directory}")
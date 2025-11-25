import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator
from functools import partial

def pubmedqa_dataloader(train_file, model_name_or_path="facebook/llama-7b", max_src_len=800, per_device_train_batch_size=8):
    # 1. 加载数据
    # 加载数据集并读取 JSON 文件
    train_dataset = load_dataset("json", data_files={'train': train_file})['train']

    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token  # 设置 pad_token 为 eos_token，以兼容 LLM 模型

    # 3. 数据预处理
    def preprocess_function(examples):
        # print("examples:{}".format(examples))
        # 直接处理每个字段，拼接成输入文本
        model_input = [
            f'Context: {support}\nQuestion: {question}\nAnswer: {correct_answer}'
            for support, question, correct_answer in zip(examples['CONTEXTS'], examples['QUESTION'], examples['final_decision'])
        ]
        
        # Tokenization
        batch = tokenizer(
            model_input,
            max_length=max_src_len,
            padding='max_length',  # 填充到最大长度
            truncation=True,  # 超过最大长度的部分截断
            add_special_tokens=True,  # 添加特殊token（如[CLS], [SEP]等，具体看模型）
            return_tensors='pt',  # 返回 PyTorch tensors
        )
        
        # 标签处理 (for training)
        batch['labels'] = batch['input_ids'][:, 1:]
        batch['input_ids'] = batch['input_ids'][:, :-1]
        batch['attention_mask'] = batch['attention_mask'][:, 1:]
        
        return batch

    
    # 4. 应用预处理函数
    train_dataset = train_dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=train_dataset.column_names,  # 移除原始的列（只保留处理过的列）
        desc="Running tokenizer on dataset"
    )

    # 5. 创建 DataLoader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=per_device_train_batch_size, 
        collate_fn=default_data_collator,  # 自动处理批次
        shuffle=True,  # 打乱数据
        pin_memory=True,  # 提高数据加载效率
    )
    
    return train_dataloader


if __name__ == "__main__":
    train_file = "../datasets/pubmedqa/pqaa_train.json"  # 训练数据文件路径
    train_dataloader = pubmedqa_dataloader(train_file, model_name_or_path="../weights/gpt-j-6b", max_src_len=800, per_device_train_batch_size=8)

    # # 现在你可以用 train_dataloader 进行训练了
    # for batch in train_dataloader:
    #     # 在这里可以处理每个 batch，进行训练
    #     print(batch)

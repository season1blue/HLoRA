import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator
from functools import partial

def piqa_dataloader(train_file, label_file, model_name_or_path="facebook/llama-7b", max_src_len=800, per_device_train_batch_size=8, num_samples=1000):
    

    # 1. 加载数据
    # 加载 JSONL 格式的训练数据
    train_dataset = load_dataset("json", data_files={'train': train_file}, split="train")

    # 2. 加载标签数据
    with open(label_file, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()]
    train_dataset = train_dataset.add_column("label", labels)

    # # 选择数量
    # train_dataset = train_dataset.select(range(num_samples))
    
    # 3. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token  # 设置 pad_token 为 eos_token，以兼容 LLM 模型

    # 4. 数据预处理
    def preprocess_function(examples, labels):
        model_input = []
        for i, example in enumerate(examples['goal']):
            goal = example
            sol1 = examples['sol1'][i]
            sol2 = examples['sol2'][i]
            label = labels[i]
            # 根据 label 选择正确的答案
            answer = sol1 if label == 0 else sol2
            model_input.append(f'Below is a goal with two proposed solutions. You need to choose one of the two solutions as the answer.Goal: {goal}\nSolution1: {sol1}; Solution2: {sol2}\nAnswer: {answer}')
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
        batch['labels'] = batch['input_ids'].clone()  # 在训练时，labels 是输入本身的一个拷贝
        # batch['labels'] = batch['input_ids'][:, 1:]
        # batch['input_ids'] = batch['input_ids'][:, :-1]
        # batch['attention_mask'] = batch['attention_mask'][:, 1:]
        
        return batch

    # 5. 应用预处理函数
    train_dataset = train_dataset.map(
        partial(preprocess_function, labels=labels),
        batched=True,
        remove_columns=train_dataset.column_names,  # 移除原始的列（只保留处理过的列）
        desc="Running tokenizer on dataset"
    )

    # 6. 创建 DataLoader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=per_device_train_batch_size, 
        collate_fn=default_data_collator,  # 自动处理批次
        shuffle=True,  # 打乱数据
        pin_memory=True,  # 提高数据加载效率
    )
    
    return train_dataloader


if __name__ == "__main__":
    train_file = "../datasets/piqa/train.jsonl"  # 训练数据文件路径
    label_file = "../datasets/piqa/train-labels.lst"  # 标签文件路径
    train_dataloader = piqa_dataloader(train_file, label_file, model_name_or_path="../weights/gpt-j-6b", max_src_len=800, per_device_train_batch_size=8)

    # 现在你可以用 train_dataloader 进行训练了
    # for batch in train_dataloader:
    #     # 在这里可以处理每个 batch，进行训练
    #     print(batch)

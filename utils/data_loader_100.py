import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset
from transformers import default_data_collator
from functools import partial

def domain_dataloader(train_file,template_name, model_name_or_path="facebook/llama-7b", max_src_len=800, per_device_train_batch_size=8):
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
        if template_name =="sciq":
            full_prompt = [
                f'Answer the following question with support.\nSupport:{support}\nQuestion:{question}\nAnswer:{correct_answer}'
                for support, question, correct_answer in zip(examples['support'], examples['question'], examples['correct_answer'])
            ]
            user_prompt = [
                f'Answer the following question with support.\nSupport:{support}\nQuestion:{question}\nAnswer:'
                for support, question in zip(examples['support'], examples['question'])
            ]
        elif template_name =="medmcqa":
            mapping_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

            full_prompt = [
                f'Solve the following medical problem by choosing the correct answer from the following four choices.\n\n### Question:\n{question}\n\n### Choices:\nA:{opa}\nB:{opb}\nC:{opc}\nD:{opd}\n\n### Answer:\n{mapping_dict[cop]}'
                for question, opa, opb, opc, opd, cop in zip(examples['question'], examples['opa'], examples['opb'], examples['opc'], examples['opd'],examples['cop'])
            ]
            user_prompt = [
                f'Solve the following medical problem by choosing the correct answer from the following four choices.\n\n### Question:\n{question}\n\n### Choices:\nA:{opa}\nB:{opb}\nC:{opc}\nD:{opd}\n\n### Answer:\n'
                for question, opa, opb, opc, opd in zip(examples['question'], examples['opa'], examples['opb'], examples['opc'], examples['opd'])
            ]

        # print(f"full_prompt:{full_prompt[1]}")
        # print(f"user_prompt:{user_prompt[1]}")

        # Tokenization
        tokenized_full_prompt = tokenizer(
            full_prompt,
            max_length=max_src_len,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,  # 添加特殊token（如[CLS], [SEP]等，具体看模型）
            return_tensors='pt',  # 返回 PyTorch tensors
        )

        tokenized_user_prompt = tokenizer(
            user_prompt,
            max_length=max_src_len,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,  # 添加特殊token（如[CLS], [SEP]等，具体看模型）
            return_tensors='pt',  # 返回 PyTorch tensors # 返回 PyTorch tensors
        )

        # 标签处理 (for training)
        tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].clone()
        tokenized_user_prompt['labels'] = tokenized_user_prompt['input_ids'].clone()

        for i,iter in enumerate(tokenized_full_prompt["labels"]):
            # 计算填充前的序列长度
            user_attention_mask = tokenized_user_prompt['attention_mask'][i]
            user_prompt_len_i = sum(user_attention_mask)

            full_attention_mask = tokenized_full_prompt['attention_mask'][i]
            full_prompt_len_i = sum(full_attention_mask)

            # print(f"tokenized_user_prompt:{user_prompt_len_i}")
            # print(f"tokenized_full_prompt:{full_prompt_len_i}")

            # print("before:",tokenized_full_prompt["labels"][i])

            # tokenized_full_prompt["labels"][i] = torch.tensor([-100]) * user_prompt_len_i + tokenized_full_prompt["labels"][i][user_prompt_len_i:]
            tokenized_full_prompt["labels"][i][:user_prompt_len_i] = -100
            
            # print("after:",tokenized_full_prompt["labels"][i])
        
        return tokenized_full_prompt

    
    # 4. 应用预处理函数
    train_dataset = train_dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=train_dataset.column_names,  # 移除原始的列（只保留处理过的列）
        desc="Running tokenizer on dataset"
    )
    # 2w datasets
    if len(train_dataset) > 20000:
        indices = list(range(20000))
        train_dataset =Subset(train_dataset,indices)

    # 5. 创建 DataLoader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=per_device_train_batch_size, 
        collate_fn=default_data_collator,  # 自动处理批次
        shuffle=True,  # 打乱数据
        pin_memory=True,  # 提高数据加载效率
    )
    
    return train_dataloader


# if __name__ == "__main__":
#     train_file = "../datasets/medmcqa/train.json"  # 训练数据文件路径
#     template_name = "medmcqa"
#     train_dataloader = domain_dataloader(train_file, template_name,model_name_or_path="../weights/gpt-j-6b", max_src_len=800, per_device_train_batch_size=8)

    # # 现在你可以用 train_dataloader 进行训练了
    # for batch in train_dataloader:
    #     # 在这里可以处理每个 batch，进行训练
    #     print(batch)

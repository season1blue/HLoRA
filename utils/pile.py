import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator
from functools import partial
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader,Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
import numpy as np



def pile_dataloader(train_file, model_name_or_path="facebook/llama-7b", max_src_len=800, per_device_train_batch_size=8, num_samples=2000):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token
    eval_dataset = load_dataset('json', data_files=train_file, split='train')
    # eval_dataset = load_dataset('json', data_files=args.val_file, split='train')

    
    def preprocess_function(examples):
        batch = tokenizer(
            examples['text'],
            max_length=max_src_len,
            padding='max_length',
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt',
        )
        # batch['labels'] = batch['input_ids'].clone()
        # batch['labels'][batch['labels'] == tokenizer.pad_token_id] = -100
        # batch['seq_len'] = batch['attention_mask'].sum(1) - 1
        batch['labels'] = batch['input_ids'][:, 1:]
        batch['input_ids'] = batch['input_ids'][:, :-1]
        batch['attention_mask'] = batch['attention_mask'][:, 1:]

        return batch

    eval_processed_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    data_total_size = len(eval_processed_dataset)

    indices = np.random.choice(data_total_size, num_samples, replace=False)
    eval_processed_dataset =Subset(eval_processed_dataset,indices)

    def data_collator_longest_padding(features):
        batch = default_data_collator(features)
        max_len = batch['attention_mask'].sum(1).max()
        for k, v in batch.items():
            if k == 'seq_len':
                continue
            batch[k] = v[:, :max_len]
        return batch

    eval_dataloader = DataLoader(
        eval_processed_dataset, collate_fn=data_collator_longest_padding, batch_size=per_device_train_batch_size, pin_memory=True
    )
    
    return eval_dataloader


if __name__ == "__main__":
    train_file = "../datasets/pile_test.jsonl"  # Pile 数据集文件路径
    train_dataloader = pile_dataloader(train_file, model_name_or_path="../weights/gpt-j-6b", max_src_len=800, per_device_train_batch_size=8)

    # 现在你可以用 train_dataloader 进行训练了
    # for batch in train_dataloader:
    #     # 在这里可以处理每个 batch，进行训练
    #     print(batch)

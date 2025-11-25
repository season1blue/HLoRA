import argparse
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss
import os
import torch
from datetime import datetime

# 设置 GPU
device = torch.device("cuda")

data_path_dict = {
    "piqa": "datasets/piqa/train.jsonl",
    "pile": "datasets/pile_test.jsonl",
    "alpaca": "datasets/alpaca_data_cleaned.json"
}
        
def save_fisher_matrix_hf_format(fisher_matrix, save_dir, config, shard_size=100):
    """
    Save Fisher matrix in Hugging Face-compatible format (shards and index.json).
    """
    os.makedirs(save_dir, exist_ok=True)
    shard = {}
    shard_index = {}
    shard_count = 0
    param_count = 0

    for n, values in fisher_matrix.items():
        shard[n] = torch.tensor(values)  # Convert values to tensor
        param_count += 1

        # When shard reaches the desired size, save it
        if param_count >= shard_size:
            shard_filename = os.path.join(save_dir, f"pytorch_model-{shard_count:05d}-of-000XX.bin")
            torch.save(shard, shard_filename)
            shard_index.update({key: os.path.basename(shard_filename) for key in shard.keys()})
            shard = {}
            param_count = 0
            shard_count += 1

    # Save any remaining parameters
    if shard:
        shard_filename = os.path.join(save_dir, f"pytorch_model-{shard_count:05d}-of-000XX.bin")
        torch.save(shard, shard_filename)
        shard_index.update({key: os.path.basename(shard_filename) for key in shard.keys()})

    # Save index file
    index_file = os.path.join(save_dir, "pytorch_model.bin.index.json")
    with open(index_file, 'w') as f:
        json.dump({
            "metadata": {"total_size": len(fisher_matrix)},
            "weight_map": shard_index
        }, f, indent=4)

    # Save config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Fisher matrix saved to Hugging Face format in {save_dir}")


def load_fisher_matrix_hf_format(save_dir):
    """
    Load Fisher matrix from Hugging Face-compatible shards.
    """
    index_file = os.path.join(save_dir, "pytorch_model.bin.index.json")
    if not os.path.exists(index_file):
        print(f"No index file found at {index_file}. Starting fresh.")
        return defaultdict(float)

    # Load index file
    with open(index_file, 'r') as f:
        shard_index = json.load(f)["weight_map"]

    fisher_matrix = {}
    for shard_filename, keys in defaultdict(list, {v: k for k, v in shard_index.items()}).items():
        shard_filepath = os.path.join(save_dir, shard_filename)
        shard = torch.load(shard_filepath)
        fisher_matrix.update(shard)

    print(f"Fisher matrix loaded from {save_dir}")
    return fisher_matrix

def get_model_input(dataset_name, index, sample):
    if dataset_name == 'alpaca':
        return f"Instruction: {sample['instruction']}\nInput: {sample['input']}\nOutput: {sample['output']}"
    elif dataset_name == 'pile':
        return sample['text']
    else:
        raise ValueError("Unsupported dataset name")
    

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime
from torch.nn import CrossEntropyLoss
import torch
import ipdb

def compute_fisher_matrix_with_dataloader(dataset_name, model_name_or_path, save_dir, batch_size, data_size, resume=False):

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto')

    data_path = data_path_dict[dataset_name]

    # Load dataset from JSONL file
    num = 0
    with open(data_path, 'r') as f:
        for line in f:
            num += 1


    # Sample a subset of data
    sampled_indices = np.random.choice(range(num), size=data_size, replace=False).tolist()
    sampled_indices = set(sampled_indices)
    samples = []
    print('Loading samples...')
    with open(data_path, 'r') as f:
        for i, line in enumerate(tqdm(f)):
            if i not in sampled_indices:
                continue
            samples.append(json.loads(line))
            if len(samples) == 20000:
                break


    print('Computing fisher matrix...')
    fisher_matrix = defaultdict(float)

    # Load model config for saving later
    config = model.config.to_dict()

    for i, batch in tqdm(enumerate(range(0, len(samples), batch_size)), total=data_size//batch_size+1):
        batch_samples = samples[batch:batch+batch_size]
        batch_samples = [sample['text'] for sample in batch_samples]
        inp = tokenizer(batch_samples, return_tensors='pt', max_length=800, truncation=True, padding='longest')
        # Move input tensors to the same device as the model (GPU)
        inp = {key: value.to(device) for key, value in inp.items()}

        labels = inp['input_ids'].masked_fill(inp['attention_mask'] == 0, -100)
        logits = model(**inp).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(logits.device)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        loss.backward()
        for n, p in model.named_parameters():
            fisher_matrix[n] = fisher_matrix[n] + (p.grad.data.detach().cpu() * len(batch_samples)) ** 2
        
        model.zero_grad()

    fisher_matrix = {n: fisher_matrix[n] / len(samples) for n in fisher_matrix}

        # # Save Fisher matrix shards periodically
        # if i % 10 == 0:  # Save every 10 batches
        #     save_fisher_matrix_hf_format(fisher_matrix, save_dir, config)
    

    # Save final Fisher matrix shards
    save_fisher_matrix_hf_format(fisher_matrix, save_dir, config)

    return fisher_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Fisher Matrix")
    parser.add_argument("--dataset_name", type=str, required=True, help="dataset_name")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pre-trained model or model name")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save/load Fisher matrix shards")
    parser.add_argument("--data_size", type=int, required=True, help="Size of data for calculated Fisher matrix")
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()

    compute_fisher_matrix_with_dataloader(
        dataset_name=args.dataset_name,
        model_name_or_path=args.model_name_or_path,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        data_size=args.data_size
    )

import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm
import numpy as np
from peft import PeftModel

norm_dict = {
    'housework_qa': 'un',
    'neg_housework_qa': 'un',
    'act_infer': 'un',
    'act_recog': 'ln',
    'count': 'ln',
    'obj_move': 'ln',
    'piqa': 'un'
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='../datasets/piqa/dev.jsonl', type=str)
    parser.add_argument('--labels_path', type=str, default='../datasets/piqa/dev-labels.lst')  # Add path for the labels file
    parser.add_argument('--model_name_or_path', type=str, default='../weights/gpt-j-6b')
    parser.add_argument('--load_lora', type=int, default=0)
    parser.add_argument('--lora_name_or_path', type=str, default='../output/lora-gptj-piqa/checkpoint')
    parser.add_argument('--output_log', action="store_true")
    parser.add_argument('--output_path', type=str, default='./eval_gptj_lora_piqa.log')
    args = parser.parse_args()
    return args

def count_lines_in_jsonl(file_path):
    with open(file_path, 'r') as f:
        line_count = sum(1 for line in f)
    return line_count

def main(args):
    norm = norm_dict[args.task_name]
    acc_list = []

    # Prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, load_in_8bit=True, device_map={"": 0})
    if args.load_lora > 0:
        model = PeftModel.from_pretrained(model, args.lora_name_or_path, device_map={"": 0})
    model.eval()

    def compute_prob(inp, contxt_len, answer_tokens):
        inputs = tokenizer(inp, return_tensors='pt')
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        logits = logits[:, contxt_len - 1:inputs['attention_mask'].sum()]
        vocab_log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            vocab_log_probs, dim=2, index=answer_tokens[:, :, None]
        )
        log_prob = token_log_probs.sum()
        return log_prob.cpu().item()

    answer_contxt_len = tokenizer('Answer:', return_tensors="pt").input_ids.size(1)

    # Load the labels from the train_label.lst file
    with open(args.labels_path, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()]

    with torch.no_grad():
        with open(args.data_path, "r") as f:
            for idx, line in tqdm(enumerate(f), total=count_lines_in_jsonl(args.data_path)):
                sample = json.loads(line)
                prompt = sample["goal"]  # The "goal" is equivalent to the "input" in the previous dataset
                answer_list = [f'solution1. {sample["sol1"]}', f'solution2. {sample["sol2"]}']  # "sol1" and "sol2" are the possible answers
                
                # Get the correct answer index from the labels file
                correct_answer_idx = labels[idx]

                # Tokenize the prompt and answer list
                prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.size(1)
                prob_list = []  # list of log prob of each answer

                for answer in answer_list:
                    answer_tokens = tokenizer(f' {answer}', return_tensors='pt').input_ids.to(model.device)
                    if norm == 'ln':
                        prob = compute_prob(f'{prompt} {answer}', prompt_len, answer_tokens)
                        final_prob = prob / answer_tokens.size(1)
                    elif norm == 'un':
                        prob = compute_prob(f'{prompt} {answer}', prompt_len, answer_tokens)
                        uncond_prob = compute_prob(f'Answer: {answer}', answer_contxt_len, answer_tokens)
                        final_prob = prob - uncond_prob
                    else:
                        raise NotImplementedError
                    prob_list.append(final_prob)

                # Now compare the predicted answer index with the correct one
                gen_idx = np.argmax(prob_list)
                acc_list.append(correct_answer_idx == gen_idx)

    # Calculate final accuracy
    acc = sum(acc_list) / len(acc_list)
    return acc

if __name__ == "__main__":
    args = parse_args()
    args.task_name = 'piqa'
    acc = main(args)

    output_str = f"{args.lora_name_or_path}: {acc:.4f}"
    print(output_str)
    if args.output_log:
        with open(args.output_path, 'a') as f:
            f.write(output_str + '\n')

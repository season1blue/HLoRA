from safetensors import safe_open
import torch

# 加载 SafeTensor 文件
with safe_open("output/rslora-gptj-medmcqa-r8/adapter_model.safetensors", framework="pt") as f:
    tensors = {}
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

# 保存为 .bin 文件
torch.save(tensors, "output/rslora-gptj-medmcqa-r8/adapter_model.bin")
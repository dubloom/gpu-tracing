import torch
print("CUDA available:", torch.cuda.is_available())
x = torch.tensor([1.0], device='cuda')
print("Tensor on GPU:", x)
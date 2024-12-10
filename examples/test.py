import torch

device = torch.cuda.current_device()
x = torch.randn(1024, 1024).to(device)
y = torch.randn(1024, 1024).to(device)
z = torch.matmul(x, y)
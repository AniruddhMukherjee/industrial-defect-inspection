import torch, torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
x = torch.randn(1000, 1000, device = device)
y = x @ x
print("matmul ran on:", y.device)
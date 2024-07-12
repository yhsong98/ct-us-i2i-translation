import torch

model = torch.load('./useg_cyclegan/latest_net_M_A.pth').keys()
print(model)
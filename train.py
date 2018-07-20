import torch

from model import Net

net = Net()
print(net)

in_data = torch.randn(1, 4, 1024, 1024)
out_data = net(in_data)
print(out_data)

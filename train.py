import torch
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from data import BratsDataset

from model import HGGNet

net = HGGNet()
data = BratsDataset(root_dir='/Users/Khedesh/Desktop/Workspace/University/Project/Data/BRATS2015_Training/output/HGG')
for i in range(len(data)):
    in_data, target = data[i]
    target = target.view(1, -1)
    criterion = CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=0.01, nesterov=True)

    # in your training loop:
    optimizer.zero_grad()  # zero the gradient buffers
    output = net(in_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()  # Does the update

from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class Net(nn.Module):
    length = 240
    length4 = length / 4

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=(1, 1))

        self.fc1 = nn.Linear(128 * self.length4 * self.length4, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        x = F.max_pool2d(
            F.relu(self.conv2(
                F.relu(self.conv2(
                    F.relu(self.conv2(
                        F.relu(self.conv1(x)))))))), (3, 3), (2, 2), padding=(1, 1))
        x = F.max_pool2d(
            F.relu(self.conv4(
                F.relu(self.conv4(
                    F.relu(self.conv4(
                        F.relu(self.conv3(x)))))))), (3, 3), (2, 2), padding=(1, 1))

        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

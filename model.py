from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    length = 240

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)

        self.fc1 = nn.Linear(128 * 80 * 80, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        x = F.max_pool2d(
            F.relu(self.conv2(
                F.relu(self.conv2(
                    F.relu(self.conv2(
                        F.relu(self.conv1(x)))))))), (3, 3), (2, 2))
        x = F.max_pool2d(
            F.relu(self.conv4(
                F.relu(self.conv4(
                    F.relu(self.conv4(
                        F.relu(self.conv3(x)))))))), (3, 3), (2, 2))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

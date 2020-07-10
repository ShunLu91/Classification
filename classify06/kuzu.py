# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.linear = nn.Linear(in_features=784, out_features=10)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return F.log_softmax(x)


class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.fc0 = nn.Sequential(nn.Linear(in_features=784, out_features=100), nn.Tanh())
        self.fc1 = nn.Sequential(nn.Linear(in_features=100, out_features=10), nn.Tanh())

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc0(x)
        x = self.fc1(x)

        return F.log_softmax(x)


class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv = nn.Sequential(
            # conv1
            nn.Conv2d(1, 32, 3, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # conv2
            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return F.log_softmax(x)

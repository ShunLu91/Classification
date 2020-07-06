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
        self.fc0 = nn.Sequential(nn.Linear(in_features=784, out_features=256), nn.Tanh())
        self.fc1 = nn.Sequential(nn.Linear(in_features=256, out_features=10), nn.Tanh())

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
        channel = [96, 144]
        # self.conv1 = nn.Sequential(nn.Conv2d(1, channel[0], 3, 1, 1), nn.BatchNorm2d(channel[0]), nn.ReLU())
        # self.conv2 = nn.Sequential(nn.Conv2d(channel[0], channel[1], 3, 1, 1), nn.BatchNorm2d(channel[1]), nn.ReLU())
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.dropout = nn.Dropout()
        # self.fc = nn.Sequential(nn.Linear(channel[1] * 7 * 7, 10), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            # nn.Linear(120, 84),
            # nn.BatchNorm1d(84),
            # nn.ReLU(),
            # nn.Linear(120, 10)
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.conv2(x)
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        # x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x)

# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_features=2, out_features=num_hid), nn.Tanh())
        self.output = nn.Sequential(nn.Linear(in_features=num_hid, out_features=1), nn.Sigmoid())

    def forward(self, input):
        r = torch.sqrt(input[:, 0] ** 2 + input[:, 1] ** 2)
        a = torch.atan2(input[:, 1], input[:, 0])
        polar = torch.stack((r, a), dim=1)
        self.hid1 = self.fc(polar)  # CHANGE CODE HERE
        output = self.output(self.hid1)

        return output


class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=num_hid)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(in_features=num_hid, out_features=num_hid)
        self.act2 = nn.Tanh()
        self.out = nn.Linear(in_features=num_hid, out_features=1)
        self.act3 = nn.Sigmoid()

    def forward(self, input):
        output = self.fc1(input)
        self.hid1 = self.act1(output)
        output = self.fc2(self.hid1)
        self.hid2 = self.act2(output)
        output = self.out(self.hid2)
        output = self.act3(output)

        return output


class ShortNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(ShortNet, self).__init__()
        self.fc1_0 = nn.Linear(in_features=2, out_features=num_hid)
        self.fc1_1 = nn.Linear(in_features=2, out_features=num_hid)
        self.fc1_2 = nn.Linear(in_features=2, out_features=1)

        self.fc2_0 = nn.Linear(in_features=num_hid, out_features=num_hid)
        self.fc2_1 = nn.Linear(in_features=num_hid, out_features=1)

        self.fc3_0 = nn.Linear(in_features=num_hid, out_features=1)

        self.act1 = nn.Tanh()
        self.act3 = nn.Sigmoid()

    def forward(self, input):
        y1_0 = self.fc1_0(input)
        y1_1 = self.fc1_1(input)
        y1_2 = self.fc1_2(input)
        self.hid1 = self.act1(y1_0)

        y2_0 = self.fc2_0(self.hid1)
        y2_1 = self.fc2_1(self.hid1)
        self.hid2 = self.act1(y1_1 + y2_0)

        y3_0 = self.fc3_0(self.hid2)
        output = self.act3(y1_2 + y2_1 + y3_0)

        return output


def graph_hidden(net, layer, node):
    if 'Polar' in str(net):
        assert layer == 1
    else:
        assert layer == 1 or layer == 2

    xrange = torch.arange(start=-7, end=7.1, step=0.01, dtype=torch.float32)
    yrange = torch.arange(start=-6.6, end=6.7, step=0.01, dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1), ycoord.unsqueeze(1)), 1)

    with torch.no_grad():  # suppress updating of gradients
        net.eval()  # toggle batch norm, dropout
        net(grid)
        if layer == 1:
            pred = (net.hid1[:, node] >= 0).float()
        elif layer == 2:
            pred = (net.hid2[:, node] >= 0).float()
        else:
            raise ValueError('wrong layer')

        # plot function computed by model
        # plt.clf()
        plt.pcolormesh(xrange, yrange, pred.cpu().view(yrange.size()[0], xrange.size()[0]), cmap='Wistia')

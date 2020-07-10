# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_features=2, out_features=num_hid), nn.Tanh())
        self.output = nn.Sequential(nn.Linear(in_features=num_hid, out_features=10), nn.Sigmoid())

    def forward(self, input):
        r = torch.sqrt(input[:, 0] ** 2 + input[:, 1] ** 2)
        a = torch.atan2(input[:, 1], input[:, 0])
        polar = torch.stack((r, a), dim=1)
        output = self.fc(polar) # CHANGE CODE HERE
        output = self.output(polar)

        return output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # INSERT CODE HERE

    def forward(self, input):
        output = 0*input[:,0] # CHANGE CODE HERE
        return output

class ShortNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(ShortNet, self).__init__()
        # INSERT CODE HERE

    def forward(self, input):
        output = 0*input[:,0] # CHANGE CODE HERE
        return output

def graph_hidden(net, layer, node):
    plt.clf()
    # INSERT CODE HERE

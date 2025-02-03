
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch import optim
from torch.utils.data import Dataset, DataLoader

INPUT_DIM = 1000

""" Light version for 3 generator case """

# class Generator(nn.Module):
    # def __init__(self):
        # super(Generator, self).__init__()
        # self.linear_0 = nn.Linear(INPUT_DIM, 1600)
        # self.conv_1 = nn.Conv2d(1, 25, kernel_size = 5)
        # self.relu_1 = nn.LeakyReLU(0.2)
        # self.conv_2 = nn.Conv2d(25, 50, kernel_size = 5)
        # self.relu_2 = nn.LeakyReLU(0.2)
        # self.conv_3 = nn.Conv2d(50, 1, kernel_size = 5)
        # self.sigm_3 = nn.Sigmoid()
    # def forward(self, x):
        # x = self.linear_0(x)
        # x = torch.reshape(x, (x.size()[0], 1, 40, 40))
        # x = self.relu_1(self.conv_1(x))
        # x = self.relu_2(self.conv_2(x))
        # out = self.sigm_3(self.conv_3(x))
        # return out


# class Discriminator(nn.Module):
    # def __init__(self):
        # super(Discriminator, self).__init__()
        # self.conv_0 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        # self.relu_0 = nn.LeakyReLU(0.2)
        # self.conv_1 = nn.Conv2d(16, 32, kernel_size=4)
        # self.relu_1 = nn.LeakyReLU(0.2)
        # self.conv_2 = nn.Conv2d(32, 64, kernel_size=3)
        # self.relu_2 = nn.LeakyReLU(0.2)
        # self.linear_2 = nn.Linear(3136, 1)
    # def forward(self, x):
        # x = self.relu_0(self.conv_0(x))
        # x = self.relu_1(self.conv_1(x))
        # x = self.relu_2(self.conv_2(x))
        # x = torch.flatten(x, start_dim=1)
        # out = self.linear_2(x)
        # return out


""" Full version (eligible in 2 generator case) """
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear_0 = nn.Linear(INPUT_DIM, 1600)
        self.conv_1 = nn.Conv2d(1, 32, kernel_size = 5)
        self.relu_1 = nn.LeakyReLU(0.2)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size = 5)
        self.relu_2 = nn.LeakyReLU(0.2)
        self.conv_3 = nn.Conv2d(64, 1, kernel_size = 5)
        self.sigm_3 = nn.Sigmoid()
    def forward(self, x):
        x = self.linear_0(x)
        x = torch.reshape(x, (x.size()[0], 1, 40, 40))
        x = self.relu_1(self.conv_1(x))
        x = self.relu_2(self.conv_2(x))
        out = self.sigm_3(self.conv_3(x))
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_0 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.relu_0 = nn.LeakyReLU(0.2)
        self.conv_1 = nn.Conv2d(16, 32, kernel_size=4)
        self.relu_1 = nn.LeakyReLU(0.2)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu_2 = nn.LeakyReLU(0.2)
        self.linear_2 = nn.Linear(3136, 1)
    def forward(self, x):
        x = self.relu_0(self.conv_0(x))
        x = self.relu_1(self.conv_1(x))
        x = self.relu_2(self.conv_2(x))
        x = torch.flatten(x, start_dim=1)
        out = self.linear_2(x)
        return out



    
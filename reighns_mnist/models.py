from typing import *

import torch
import torch.nn.functional as F
from torch import nn, optim


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # in_channels is 1 for first conv layer due to grayscale image, 1 channel only.
        conv_layers: Dict = {'conv1': {'in_channels': 1, 'out_channels': 10, 'kernel_size': 5,
                                       'stride': 1, 'padding': 0, 'bias': True, 'padding_mode': 'zeros',
                                       'device': None, 'dtype': None},
                             'conv2': {'in_channels': 10, 'out_channels': 20, 'kernel_size': 5,
                                       'stride': 1, 'padding': 0, 'bias': True, 'padding_mode': 'zeros',
                                       'device': None, 'dtype': None},
                             'fc1': {'in_features': 320, 'out_features': 50, 'bias': True,
                                     'device': None, 'dtype': None},
                             'fc2': {'in_features': 50, 'out_features': 10, 'bias': True,
                                     'device': None, 'dtype': None},
                             }

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
        # return F.log_softmax(x, dim=0)

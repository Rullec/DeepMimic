import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


class fc_2layers_128_64(nn.Module):
    NAME = "fc_2layers_128_64"

    def __init__(self, input_size, output_size, activation):
        assert type(activation) == str
        super(fc_2layers_128_64, self).__init__()

        if activation == "leaky_relu":
            self.activation = F.leaky_relu
        elif activation == "relu":
            self.activation = F.relu
        else:
            assert False, f"unsupported activation {activation}"

        # define layers
        self.input = nn.Linear(input_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.input(x)
        x = self.activation(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x

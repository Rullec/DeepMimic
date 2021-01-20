import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


class fc_2layers_64_16(nn.Module):
    NAME = "fc_2layers_64_16"

    def __init__(self, input_size, output_size, activation):
        assert type(activation) == str
        super(fc_2layers_64_16, self).__init__()

        # define layers
        if activation == "leaky_relu":
            self.activation = F.leaky_relu
        elif activation == "relu":
            self.activation = F.relu
        else:
            assert False, f"unsupported activation {activation}"
        self.input = nn.Linear(input_size, 64)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, output_size)

    def forward(self, x):
        x = self.input(x)
        x = self.activation(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x

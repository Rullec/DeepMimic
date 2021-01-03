import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


class fc_2layers_128_64(nn.Module):
    NAME = "fc_2layers_128_64"

    def __init__(self, input_size, output_size):
        super(fc_2layers_128_64, self).__init__()

        # define layers
        self.input = nn.Linear(input_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        # concatenate layers
        # x = F.relu(self.input(x))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.input(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

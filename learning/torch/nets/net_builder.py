import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


class fc_layers(nn.Module):

    def __init__(self, name, input_size, output_size, layer_sizes, activation, dropout):
        super(fc_layers, self).__init__()
        self.name = name
        assert type(activation) == str
        dropout = 0 if dropout == None else dropout
        assert type(dropout) is float
        assert 1 >= dropout >= 0
        assert type(layer_sizes) is list
        for i in layer_sizes:
            assert(type(i)) is int

        if len(layer_sizes) == 0:
            layer_sizes.append(output_size)
        # define layers
        if activation == "leaky_relu":
            self.activation = F.leaky_relu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "none":
            self.activation = None
        else:
            assert False, f"unsupported activation {activation}"
        print(f"[debug] dropout = {dropout}")
        self.dropout_layer = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList()
        for _idx in range(len(layer_sizes) + 1):
            if _idx == 0:
                self.layers.append(
                    nn.Linear(input_size, layer_sizes[_idx]))
            elif _idx == len(layer_sizes):
                self.layers.append(
                    nn.Linear(layer_sizes[_idx - 1], output_size))
            else:
                # no dropout and activation in the last layer
                self.layers.append(
                    nn.Linear(layer_sizes[_idx - 1], layer_sizes[_idx]))

    def forward(self, x):
        for _idx, layer in enumerate(self.layers):
            x = layer(x)
            # add activation if not the final layer
            if _idx != len(self.layers) - 1:
                if self.activation is not None:
                    x = self.activation(x)
            # add dropout
            x = self.dropout_layer(x)

        return x


def build_net(net_name, input_size, output_size, activation, dropout):
    layer_sizes = [int(i) for i in net_name.split('_')[2:]]

    net = fc_layers(net_name, input_size, output_size,
                    layer_sizes, activation, dropout)

    # print(f"net param {net.parameters()}")
    # print(f"net param {list(net.state_dict().keys())}")
    return net

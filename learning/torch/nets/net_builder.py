from learning.torch.nets.fc_2layers_128_64 import fc_2layers_128_64
from learning.torch.nets.fc_2layers_64_16 import fc_2layers_64_16


def build_net(net_name, input_size, output_size, activation):
    net = None
    if net_name == fc_2layers_128_64.NAME:
        net = fc_2layers_128_64(input_size, output_size, activation)
    if net_name == fc_2layers_64_16.NAME:
        net = fc_2layers_64_16(input_size, output_size, activation)
    else:
        assert False, f"unsupported net type {net_name}"
    return net

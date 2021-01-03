from learning.torch.nets.fc_2layers_128_64 import fc_2layers_128_64


def build_net(net_name, input_size, output_size):
    net = None
    if net_name == fc_2layers_128_64.NAME:
        net = fc_2layers_128_64(input_size, output_size)
    else:
        assert False, f"unsupported net type {net_name}"
    return net

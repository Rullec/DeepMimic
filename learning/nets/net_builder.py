import learning.nets.fc_2layers_1024units as fc_2layers_1024units

def build_net(net_name, input_tfs, reuse=False):
    net = None

    # 建立网络，给定输入: 如果这个网络的名字是fc_2layers_1024units
    if (net_name == fc_2layers_1024units.NAME):
        net = fc_2layers_1024units.build_net(input_tfs, reuse)
    else:
        assert False, 'Unsupported net: ' + net_name
    
    return net
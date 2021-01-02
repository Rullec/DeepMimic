import learning.tf.nets.fc_2layers_1024units as fc_2layers_1024units
import learning.tf.nets.fc_2layers_256_128 as fc_2layers_256_128
import learning.tf.nets.fc_2layers_512_128 as fc_2layers_512_128
import learning.tf.nets.fc_2layers_512_256 as fc_2layers_512_256
import learning.tf.nets.fc_3layers_128_128_64 as fc_3layers_128_128_64
import learning.tf.nets.fc_2layers_128_64 as fc_2layers_128_64
import learning.tf.nets.fc_2layers_256_256 as fc_2layers_256_256


def build_net(net_name, input_tfs, reuse=False):
    net = None

    if (net_name == fc_2layers_1024units.NAME):
        net = fc_2layers_1024units.build_net(input_tfs, reuse)
    elif(net_name == fc_2layers_256_128.NAME):
        net = fc_2layers_256_128.build_net(input_tfs, reuse)
    elif(net_name == fc_2layers_256_256.NAME):
        net = fc_2layers_256_256.build_net(input_tfs, reuse)
    elif(net_name == fc_2layers_512_256.NAME):
        net = fc_2layers_512_256.build_net(input_tfs, reuse)
    elif(net_name == fc_2layers_512_128.NAME):
        net = fc_2layers_512_128.build_net(input_tfs, reuse)
    elif(net_name == fc_3layers_128_128_64.NAME):
        net = fc_3layers_128_128_64.build_net(input_tfs, reuse)
    elif(net_name == fc_2layers_128_64.NAME):
        net = fc_2layers_128_64.build_net(input_tfs, reuse)
    else:
        assert False, 'Unsupported sa: ' + net_name

    return net

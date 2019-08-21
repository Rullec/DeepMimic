import tensorflow as tf
import learning.tf_util as TFUtil

NAME = "fc_2layers_1024units"

def build_net(input_tfs, reuse=False):
    layers = [1024, 512]
    activation = tf.nn.relu
    # 输出是0 - 1之间

    # 输入 -> relu(1024) -> relu(512) -> 输出
    input_tf = tf.concat(axis=-1, values=input_tfs)          
    h = TFUtil.fc_net(input_tf, layers, activation=activation, reuse=reuse)
    h = activation(h)   # relu是0 - Inf
    return h
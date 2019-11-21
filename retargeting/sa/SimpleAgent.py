from typing import Optional, List, Any, Union, Tuple

from learning.tf_agent import TFAgent
import learning.tf_util as TFUtil
from util.io import load_buffer_npz
import numpy as np
import tensorflow as tf
from retargeting.sa.SimpleAgentNetWork import NetWork


class SimpleAgent(TFAgent):
    NAME = 'SA'
    BUFFER_PATH = 'buffer_path'
    STATE_MEAN_PATH = 'mean_path'
    STATE_STD_PATH = 'std_path'
    NET_PATH = 'net_save_path'

    def __init__(self, world, id, json_data):
        self.network_initialized = False
        self.json_data = json_data
        self.lr = 1e-3
        self.lamda = 1e-3
        super().__init__(world, id, json_data)
        self.buffers = None
        self.init_agent(json_data)

    def _build_normalizers(self):
        mean_path = self.json_data[self.STATE_MEAN_PATH]
        std_path = self.json_data[self.STATE_STD_PATH]
        self.state_std = np.load(std_path)['std']
        self.state_mean = np.load(mean_path)['mean']

    def _norm(self, s):
        return (s - self.state_mean) / self.state_std

    def _build_bounds(self):
        if self.world is not None:
            super()._build_bounds()

    def _build_losses(self, json_data):
        # self.global_step = tf.Variable(0, trainable=False)
        # self.learning_rate = tf.train.exponential_decay(self.lr,
        #                                                 global_step=self.global_step,
        #                                                 decay_steps=10000,
        #                                                 decay_rate=0.99,
        #                                                 staircase=True)
        #
        total_vars = tf.trainable_variables()
        weights_name_list = [var for var in total_vars if "kernel" in var.name]
        loss_holder = []
        #
        for w in range(len(weights_name_list)):
            l2_loss = tf.nn.l2_loss(weights_name_list[w])
            loss_holder.append(l2_loss)

        # self.regular_loss = tf.reduce_mean(loss_holder) * self.lamda
        # self.regression_loss = tf.losses.mean_squared_error(self.a_out, self.a_tf)
        # self.loss = self.regression_loss + self.regular_loss
        # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _build_solvers(self, json_data):
        pass

    def _decide_action(self, s, g):
        s = np.reshape(s, [1, len(s)])
        s = self._norm(s)
        return self.net.eval(s)

    def _train_step(self):
        pass

    def _check_action_space(self):
        return True

    def _build_nets(self, json_data):
        s_size = self.get_state_size()
        a_size = self.get_action_size()

        self.net = NetWork(s_size, a_size)

        # layers = [1024, 512, 256, 128]
        # activation = tf.nn.relu
        # self.s_tf = tf.placeholder(tf.float32, shape=[None, s_size], name="s")  # 输入state
        # self.a_tf = tf.placeholder(tf.float32, shape=[None, a_size], name="a")  # 输入action
        # self.fc = TFUtil.fc_net(input=self.s_tf, layers_sizes=layers, activation=activation)
        # h = activation(self.fc)
        # self.a_out = tf.layers.dense(inputs=h, units=a_size, activation=None,
        #                              kernel_initializer=tf.random_uniform_initializer(minval=-1, maxval=1))


    def init_agent(self, json_data):
        buffer_dir = json_data[self.BUFFER_PATH]
        self._init_normalizers()

    def _build_saver(self):
        self.saver = tf.train.Saver()
        net_dir = self.json_data[self.NET_PATH]

    def _init_normalizers(self):
        if self.network_initialized:
            super()._init_normalizers()

    def _load_data(self, buffer_dir):
        self.buffers = load_buffer_npz(buffer_dir)
        self.convert_buffer_format()

    def convert_buffer_format(self):
        print('len of buffer: {}'.format(len(self.buffers)))
        states = []
        actions = []
        for b in self.buffers:
            try:
                state  = b['states']
                action = b['actions']
            except Exception as e:
                continue
            states.append(state)
            actions.append(action)
        states = np.array(states)
        actions = np.array(states)
        print(states[90].shape)

    def test(self):
        assert self.buffers is not None
        b = self.buffers[0]


import tensorflow as tf
import learning.tf_util as TFUtil
import numpy as np
import datetime
import os

device_name = '/cpu:0'


class NetWork(object):
    def __init__(self, input_dim, output_dim, lr=3e-4, batch_size=200):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.n_epoch = 1000
        self.batch_size = batch_size
        self.lamda = 1e-3
        self._build_net()
        self._build_losses()
        self._build_saver()
        self.save_step = 100
        self.state_mean = 0
        self.state_std = 0

    def _build_net(self):
        layers = [1024, 512, 256, 128]
        activation = tf.nn.relu
        with tf.device(device_name):
            self.s_tf = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="s")  # 输入state
            self.a_tf = tf.placeholder(tf.float32, shape=[None, self.output_dim], name="a")  # 输入action
            self.fc = TFUtil.fc_net(input=self.s_tf, layers_sizes=layers, activation=activation)
            h = activation(self.fc)
            self.a_out = tf.layers.dense(inputs=h, units=self.output_dim, activation=None,
                                         kernel_initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

    def _build_losses(self):
        self.sess = tf.Session()

    def _build_saver(self):
        self.saver = tf.train.Saver()
        with self.sess.as_default():
            filename = '/data/ubuntu_data/DeepMimic/output/1120/model/sa_net.ckpt-1000'
            self.saver.restore(self.sess, filename)

    def _create_data_set(self, feature, label):
        data = tf.data.Dataset.from_tensor_slices((feature, label))
        data = data.repeat()
        # data = data.shuffle(10000)
        data = data.batch(batch_size=self.batch_size)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        iterator = data.make_initializable_iterator()
        return iterator

    def _init_data_set(self, state, action, portion):
        '''
            split data and create iterator
        '''
        train_size = int(state.shape[0] * portion)
        test_size = state.shape[0] - train_size
        train_state, train_action = state[:train_size], action[:train_size]
        test_state, test_action = state[train_size:], action[train_size:]
        train_iter = self._create_data_set(feature=train_state, label=train_action)
        test_iter = self._create_data_set(feature=test_state, label=test_action)
        return train_iter, test_iter

    def eval(self, state):
        with self.sess.as_default():
            a = self.sess.run(self.a_out, feed_dict={self.s_tf: state})
            # a_0 = np.zeros_like(a[0])
        return a[0], 0.1

    def save_normalizer(self, path):
        mean_path = os.path.join(path, 'mean.npz')
        std_path = os.path.join(path, 'std.npz')
        np.savez_compressed(mean_path, mean=self.state_mean)
        np.savez_compressed(std_path, std=self.state_std)

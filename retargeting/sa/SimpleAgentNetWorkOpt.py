import tensorflow as tf
import learning.tf.tf_util as TFUtil
import numpy as np
import datetime
import os

device_name = '/cpu:0'


class NetWork(object):
    def __init__(self, input_dim, output_dim, json_data):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = json_data['layers']
        self.net_save_path = json_data['net_save_path']

        self._build_net()
        self._build_losses()
        self._build_saver()

    def _build_net(self):
        layers = self.layers
        activation = tf.nn.relu
        with tf.device(device_name):
            self.s_tf = tf.placeholder(
                tf.float32, shape=[None, self.input_dim], name="s")
            self.a_tf = tf.placeholder(
                tf.float32, shape=[None, self.output_dim], name="a")
            self.fc = TFUtil.fc_net(
                input=self.s_tf, layers_sizes=layers, activation=activation)
            h = activation(self.fc)
            self.a_out = tf.layers.dense(inputs=h, units=self.output_dim, activation=None,
                                         kernel_initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

    def _build_losses(self):
        self.sess = tf.Session()

    def _build_saver(self):
        self.saver = tf.train.Saver()
        with self.sess.as_default():
            print("[log] SimpleAgentNetWorkOpt load mode from %s" %
                  self.net_save_path)
            self.saver.restore(self.sess, self.net_save_path)

    def eval(self, state):
        with self.sess.as_default():
            a = self.sess.run(self.a_out, feed_dict={self.s_tf: state})
        return a[0], 0.1, a[0]

import numpy as np
import tensorflow as tf

from learning import tf_util as TFUtil
from shapevar.shapegen.shape_gen import ShapeGen
from learning.tf_normalizer import TFNormalizer

np.random.seed(1)
device_name = '/cpu:0'


class MCMCShapeGen(ShapeGen):
    NAME = 'MCMC'
    val_norm = None

    def __init__(self, shape_dim, shape_lb, shape_ub, nn_layers, l2_coeff, proposal, lr_nn, lr_mu, k):
        super().__init__(shape_dim, shape_lb, shape_ub, nn_layers, l2_coeff, proposal, lr_nn, lr_mu, k)
        self.buffer_limit = 1000
        self.buffer = {'shape': [], 'v:': []}
        self.norm_mean = 0
        self.norm_std = 1
        self.update_itr = 0
        self.current_lr = lr_nn
        self.lr_decay_rate = 1
        self.lr_decay_steps = 1e+5

    def init_network(self):
        self._build_network()
        self._build_loss()
        self._init_tf()
        self._init_uniform_prob()

    def _build_network(self):
        assert self.shape_dim != 0
        layers = self.nn_layers
        activation = tf.nn.relu
        with tf.device(device_name):
            # body shape param input
            self.sb_tf = tf.placeholder(tf.float32, shape=[None, self.shape_dim], name="sb")
            # vsb_ means \bar V(sb)
            self.V_sb_tf = tf.placeholder(tf.float32, shape=[None, 1], name="vsb_")
            self.fc = TFUtil.fc_net(input=self.sb_tf, layers_sizes=layers, activation=activation)
            h = activation(self.fc)
            self.V_sb_out_norm = tf.layers.dense(inputs=h, units=1, activation=activation,
                                                 kernel_initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

            self.V_sb_out = self.V_sb_out_norm * self.norm_std + self.norm_mean

    def _build_loss(self):
        # using regression loss for predict body shape marginal value function
        self.current_lr_tf = tf.placeholder(dtype=tf.float32)
        v_sb_tf_norm = (self.V_sb_tf - self.norm_mean) / self.norm_std
        v_sb_out_norm = (self.V_sb_out - self.norm_mean) / self.norm_std
        norm_val_diff = v_sb_tf_norm - v_sb_out_norm
        self.regression_loss = 0.5 * tf.reduce_mean(tf.square(norm_val_diff))
        # self.regression_loss = tf.reduce_mean(tf.square(self.V_sb_tf - self.V_sb_out))

        total_vars = tf.trainable_variables()
        weights_name_list = [var for var in total_vars if "kernel" in var.name]
        loss_holder = []
        for w in range(len(weights_name_list)):
            l2_loss = tf.nn.l2_loss(weights_name_list[w])
            loss_holder.append(l2_loss)
        # using regular loss for preventing over fitting
        self.regular_loss = tf.reduce_mean(loss_holder) * self.l2_coeff
        self.loss = self.regression_loss + self.regular_loss
        self.optimizer = tf.train.AdamOptimizer(self.current_lr_tf)
        self.train_op = self.optimizer.minimize(self.loss)

    def _init_tf(self):
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _init_uniform_prob(self):
        length = self.shape_ub - self.shape_lb
        self.u = 1
        for d in length:
            self.u *= 1 / d

    def _evaluate_stage(self, sb, sb_prime):
        if len(np.shape(sb)) == 1:
            sb = np.reshape(sb, [1, len(sb)])
        v_sb = self.sess.run(self.V_sb_out, feed_dict={self.sb_tf: sb})
        v_sb_prime = self.sess.run(self.V_sb_out, feed_dict={self.sb_tf: sb_prime})
        p_sb = np.exp(-self.k * (v_sb - self.mu) / self.mu)
        p_sb_prime = np.exp(-self.k * (v_sb_prime - self.mu) / self.mu)
        return p_sb_prime / p_sb

    def _reject_stage(self, sb, sb_prime, alpha):
        if self.proposal == ShapeGen.UNIFORM_DISTRIBUTION:
            if np.random.random() < np.squeeze(alpha):
                return sb_prime
            else:
                return sb
        else:
            print("Error, Please Correct Proposal Function!")
            exit(-1)

    def generate_shape(self, sb_input):
        """sample next shape param at current shape"""
        sb_prime = self._generate_stage()
        alpha = self._evaluate_stage(sb_input, sb_prime)
        return self._reject_stage(sb_input, sb_prime, alpha)

    def update(self, sb_input, v_target):
        """train vsb network using data from buffer"""
        # update vsb nn
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.current_lr_tf: self.current_lr, self.sb_tf: sb_input, self.V_sb_tf: v_target})
        # update mu
        delta_mu = np.sum(self.mu - v_target, axis=0)
        self.mu -= delta_mu * self.lr_mu
        self.update_itr += 1
        self.update_learning_rate()
        return loss

    def store_pair(self, sb, tar_val):
        pass

    def update_learning_rate(self):
        self.current_lr = self.lr_nn * self.lr_decay_rate ** (self.update_itr / self.lr_decay_steps)


def Test():
    shape_ub = np.array([1.5, 1.6])
    shape_lb = np.array([0.5, 0.4])
    shape_dim = len(shape_lb)
    layers = [256, 128, 64]
    l2_coeff = 1e-4
    lr_nn = 1e-3
    lr_mu = 1e-2
    k = 1
    gen = ShapeGen(shape_dim, shape_lb=shape_lb, shape_ub=shape_ub, nn_layers=layers, l2_coeff=l2_coeff,
                   proposal=ShapeGen.UNIFORM_DISTRIBUTION, lr_mu=lr_mu, lr_nn=lr_nn, k=k)
    sb0 = np.array([[1, 1]])
    sb = sb0
    buffer = []
    thresh = 50
    for i in range(30000):
        sb = gen.generate_shape(sb)
        buffer.append(sb)
        # print(sb)
        if len(buffer) == thresh:
            v_target = []
            for s in buffer:
                v_target.append([np.squeeze((s - sb0).dot((s - sb0).T))])
            gen.update(np.vstack(buffer), np.array(v_target))
            buffer.clear()

    buffer.clear()
    for _ in range(100):
        sb = gen.generate_shape(sb)
        buffer.append(sb)
        print(sb)
    buffer = np.vstack(buffer)
    v = 0
    for b in buffer:
        v += (b - sb0).dot((b - sb0).T)
    print(v)


if __name__ == '__main__':
    Test()

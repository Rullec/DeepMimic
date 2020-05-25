import tensorflow as tf
import learning.tf_util as TFUtil
import numpy as np
import datetime
import os

device_name = '/cpu:0'


class NetWork(object):
    def __init__(self, input_dim, output_dim, json_data):
        self.input_dim  = input_dim
        self.output_dim = output_dim

        self.lr         = json_data['lr']
        self.n_epoch    = json_data['n_epoch']
        self.batch_size = json_data['batch_size']
        self.l2_coeff   = json_data['l2_coeff']
        self.output_dir = json_data['output_path']
        self.save_step  = json_data['save_step']
        self.layers     = json_data['layers']
        self.json_data  = json_data

        self._build_net()
        self._build_losses()
        self._build_saver()
        self.state_mean = 0
        self.state_std  = 0

    def _build_net(self):
        layers = self.layers
        activation = tf.nn.relu
        with tf.device(device_name):
            self.s_tf = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="s")
            self.a_tf = tf.placeholder(tf.float32, shape=[None, self.output_dim], name="a")
            self.fc = TFUtil.fc_net(input=self.s_tf, layers_sizes=layers, activation=activation)
            h = activation(self.fc)
            self.a_out = tf.layers.dense(inputs=h, units=self.output_dim, activation=None,
                                         kernel_initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

    def _build_losses(self):
        # self.global_step = tf.Variable(0, trainable=False)
        # self.learning_rate = tf.train.exponential_decay(self.lr,
        #                                                 global_step=self.global_step,
        #                                                 decay_steps=10000,
        #                                                 decay_rate=0.99,
        #                                                 staircase=True)

        total_vars = tf.trainable_variables()
        weights_name_list = [var for var in total_vars if "kernel" in var.name]
        loss_holder = []

        for w in range(len(weights_name_list)):
            l2_loss = tf.nn.l2_loss(weights_name_list[w])
            loss_holder.append(l2_loss)

        self.regular_loss = tf.reduce_mean(loss_holder) * self.l2_coeff
        self.regression_loss = tf.losses.mean_squared_error(self.a_out, self.a_tf)
        self.loss = self.regression_loss + self.regular_loss
        # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        # tf.summary.scalar('regression_loss', self.regression_loss)
        # tf.summary.scalar('regular_loss', self.regular_loss)
        # tf.summary.scalar("loss", self.loss)
        # tf.summary.scalar("lr", self.learning_rate)

        self.merged_summary_op = tf.summary.merge_all()
        self.sess = tf.Session()

    def _build_saver(self):
        self.saver = tf.train.Saver()
        with self.sess.as_default():
            filename = self.json_data['net_save_path']
            # assert(os.path.exists(filename))
            # filename = '/home/ljf/playground/TF/sa/output/exp-1/model/sa_net.ckpt-1000'
            print("[log] SimpleAgentNetWorkOpt load mode from %s" % filename)
            self.saver.restore(self.sess, filename)
            # exit(0)

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

    def train(self, state, action):
        print("start training")
        self.state_mean = state.mean(axis=0)
        self.state_std = state.std(axis=0)
        for i in np.where(self.state_std == 0):
            self.state_std[i] = 1
        state = (state - self.state_mean) / self.state_std

        norm_path = os.path.join(self.output_dir, 'norm')

        self.save_normalizer(norm_path)
        init = tf.global_variables_initializer()
        train_iter, test_iter = self._init_data_set(state, action, 0.7)
        train_next = train_iter.get_next()
        test_next = test_iter.get_next()
        logs_dir_name = 'log'
        logs_path = os.path.join(self.output_dir, logs_dir_name)

        with tf.Session() as sess:
            sess.run(init)
            summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
            sess.run(train_iter.initializer)
            sess.run(test_iter.initializer)
            # filename = './model/sa_net.ckpt-1000'
            # self.saver.restore(sess, filename)

            train_size = int(state.shape[0] * 0.7)
            test_size = state.shape[0] - train_size

            n_batch = state.shape[0] // self.batch_size
            n_epoch = self.n_epoch

            train_batches = range(n_batch)[:int(0.7 * n_batch)]
            test_batches = range(n_batch)[int(0.7 * n_batch):]

            total_step = 0
            train_step = 0

            for e in range(n_epoch):
                train_loss = 0
                train_regression_loss = 0
                for i in train_batches:
                    x, y = sess.run(train_next)
                    _, regression, regular, summary = sess.run([self.train_op, self.regression_loss, self.regular_loss, self.merged_summary_op], feed_dict={self.s_tf: x, self.a_tf: y, self.global_step: train_step})
                    summary_writer.add_summary(summary, train_step)
                    train_loss += regression + regular
                    train_regression_loss += regression
                    total_step += 1
                    train_step += 1
                print('Epoch: {}, Train Loss: {}, regresion loss: {}'.format(e, train_loss/len(train_batches), train_regression_loss/len(train_batches)))
                test_loss = 0
                test_regression = 0
                for i in test_batches:
                    x, y = sess.run(test_next)
                    regression, regular = sess.run([self.regression_loss, self.regular_loss], {self.s_tf: x, self.a_tf: y})
                    test_loss += regression + regular
                    test_regression += regression
                    total_step += 1
                print('Epoch: {}, Test  Loss: {}, regresion loss: {}'.format(e, test_loss/len(test_batches), test_regression/len(test_batches)))
                if (e + 1) % self.save_step == 0:
                    filename = 'model/sa_net.ckpt'
                    model_save_path = os.path.join(self.output_dir, filename)
                    self.saver.save(sess, model_save_path, global_step=(e + 1))

    def eval(self, state):
        with self.sess.as_default():
            a = self.sess.run(self.a_out, feed_dict={self.s_tf: state})
            # a_0 = np.zeros_like(a[0])
        return a[0], 0.1, a[0]

    def save_normalizer(self, path):
        mean_path = os.path.join(path, 'mean.npz')
        std_path = os.path.join(path, 'std.npz')
        np.savez_compressed(mean_path, mean=self.state_mean)
        np.savez_compressed(std_path, std=self.state_std)


import sys,os,shutil,pickle
import tensorflow as tf
import numpy as np
import json
from operator import itemgetter
import matplotlib.pyplot as plt
import argparse

weight_name_info = ['agent/main/actor/0/dense/kernel:0', 'agent/main/actor/0/dense/bias:0', 'agent/main/actor/1/dense/kernel:0',
 'agent/main/actor/1/dense/bias:0', 'agent/main/actor/dense/kernel:0', 'agent/main/actor/dense/bias:0']
weight_shape_info = [(275, 1024), (1024,), (1024, 512), (512,), (512, 58), (58,)]
project_dir = os.getcwd()
default_info_file_path = os.path.join(project_dir, "data0824/models/info.json")


def select_key_pts(i, conf_lst,  k = 5):
    '''
        给出所有configuraion info的列表
        返回第i个的k=5近邻的id
        conf_lst = []
    '''
    conf_mat = np.array(conf_lst)   # 要求conf_mat.shape = (links_num, skeleton_num)
    # print("conf_mat shape = %s" % str(conf_mat.shape))
    
    target_vec = np.reshape(np.tile(conf_mat[i], conf_mat.shape[0]), conf_mat.shape)# 将conf_mat[i]，也就是第i个conf重复70次，也形成一个70*18的矩阵
    
    norm2_res = np.linalg.norm(conf_mat - target_vec, axis=1)   # 两个矩阵相减，并且对axis=1求l2-norm，得到一个70*1个列表，这就是各个conf到第i个conf的距离。
    norm2_res_id = np.vstack([np.arange(0, len(conf_lst), 1), norm2_res])  # 给上面那个70*1的列表，附加上一个0-69的id信息，形成一个矩阵为2*70, 其中第一行为0-69的顺序id, 第二行为对应的距离
    
    order = np.lexsort(norm2_res_id)    # 以第二行的数字为关键值，对这个2*70数组的70个列进行排序
    sorted_norm2 = norm2_res_id[:,order]
    
    # 获取距离最近的前k个id:
    id_res = sorted_norm2[0, 1:k+1]
    dist_res = sorted_norm2[1, 1:k+1]
    return id_res.astype(int).tolist(), dist_res.tolist()

def load_single_skeleton(filename):
    body_value = None
    with open(filename) as f:
        body_value = json.load(f)["BodyDefs"]
    assert body_value is not None
    link_name_lst = ["root", "rhipjoint", "rfemur", "rtibia", "rfoot", "lowerback", "upperback", "thorax",
                    "rclavicle", "rhumerus", "rradius", "rwrist", "lowerneck", "lclavicle", "lhumerus", 
                    "lradius", "lwrist", "lhipjoint", "lfemur", "ltibia", "lfoot"]
    conf = np.zeros(len(link_name_lst))
    for i in body_value:
        id = i["ID"]
        name = i["Name"]
        assert name == link_name_lst[id]
        length = i["Param1"]
        conf[id] = length
        
    return conf

def read_train_data_info(filepath):
    # print("info path = %s" % filepath)
    weight_lst = []
    skeleton_lst = []
    with open(filepath, "r") as f:
        value = json.load(f)
        for i in value:
            weight_lst.append(i["weight"])
            skeleton_lst.append(i["skeleton"])
    # print(weight_lst)
    # print(skeleton_lst)
    return weight_lst, skeleton_lst

def sample(theta_mat, conf_lst, K, batch_size = 8):
    '''
        theta_mat.shape == (sample_num, sample_dim) = (25个样本，每个样本1000维度)
        conf_lst = [1st conf, 2nd conf, ..., nth conf]
    '''
    theta_num = theta_mat.shape[1]

    # 采样batchsize个数据
    train_data_total = None
    target_data_total = None
    weight_data_total = None
    center_pt_id = []
    for _ in range(batch_size):
        # 采样一个数据
        i = np.random.randint(len(conf_lst))
        center_pt_id.append(i)
        pt_id_lst, pt_dist_list = select_key_pts(i, conf_lst,  k=K)

        train_data = theta_mat[pt_id_lst]
        target_data = theta_mat[i]

        # 计算conf之间的权重: 是距离的反比，然后归一化(变成和为1)
        weight_data = np.array([1.0/i for i in pt_dist_list])
        weight_data /= sum(weight_data)
        # print("weights: %s" % str(weights))
        train_data = np.reshape(train_data, (-1, K, theta_num))
        target_data = np.reshape(target_data, (-1, 1, theta_num))
        weight_data = np.reshape(weight_data, (-1, 1, K))

        # 把当前data合并到total里面
        merge = lambda total, item: item if total is None else np.vstack([total, item])
        train_data_total = merge(train_data_total, train_data)
        target_data_total = merge(target_data_total, target_data)
        weight_data_total = merge(weight_data_total, weight_data)
    
    assert train_data_total.shape == (batch_size, K, theta_num)
    assert target_data_total.shape == (batch_size, 1, theta_num)
    assert weight_data_total.shape == (batch_size, 1, K)
    return train_data_total, target_data_total, weight_data_total, center_pt_id

def read_batch_weight(weight_path_lst, weight_name_lst, weight_shape_lst):
    weight_mat = None
    for weight_path in weight_path_lst:
        if os.path.exists(weight_path) == False:
            raise(FileNotFoundError)
        with open(weight_path, "rb") as f:
            weight_dict = pickle.load(f)
            assert len(weight_dict) == len(weight_name_lst)

            weight_vec = np.zeros(0)
            # print("**********************")
            for i in weight_name_lst:
                # print("raw: %s " % str(weight_dict[i].shape))
                item = np.reshape(weight_dict[i],(-1,))
                # print("after: %s " % str(weight_dict[i].shape))
                weight_vec = np.hstack([weight_vec, item])
            if weight_mat is None:
                weight_mat = weight_vec
            else:
                weight_mat = np.vstack([weight_mat, weight_vec])
    # print(weight_mat.shape)
    return weight_mat

def single_theta_vec2dict(theta_vec, weight_name_lst, weight_shape_lst):
    assert len(weight_name_lst) == len(weight_shape_lst)
    get_num = lambda tup: tup[0] if len(tup) == 1 else tup[0] * tup[1]

    # 获取shape中各个tuple所代表的参数个数
    theta_num_lst = [get_num(i) if len(i)<=2  else AssertionError for i in weight_shape_lst]
    theta_num = np.sum(theta_num_lst)

    assert theta_vec.shape == (1, 1, theta_num)
    theta_vec = np.reshape(theta_vec, (theta_num, ))    # 变形，准备数据

    # 从theta_vec中顺序取出元素，
    theta_dict = {}
    cur_ptr = 0
    for i in range(len(weight_name_lst)):
        name = weight_name_lst[i]
        theta_dict[name] = np.reshape(theta_vec[cur_ptr: cur_ptr+theta_num_lst[i]], weight_shape_lst[i])
        cur_ptr += theta_num_lst[i]
    assert cur_ptr == theta_num
    
    return theta_dict

class autoencoder:
    def __init__(self, theta_num = 1000, lr = 1e-3, lr_decay=0.95, lr_decay_epoches=10, lr_min=1e-6):
        
        self.para_init(theta_num, lr, lr_decay, lr_decay_epoches, lr_min)

        self.epoch = 0
        self.iter = 0
        self.total_iter = 0
        self.build_network()

    def para_init(self, theta_num, lr, lr_decay, lr_decay_epoches, lr_min):
        # 因为不同算法的参数不一样，所以这部分必须独立实现。
        self.hyper_para_list = ["theta_num", "lr"]
        self.param_num = len(self.hyper_para_list)
        self.theta_num = theta_num
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_epoches = lr_decay_epoches
        self.lr_min = lr_min
        self.loss1_weight = 1.      # 插值controller恢复的loss
        self.loss2_weight = 1       # target controller恢复的loss
        self.loss3_weight = 100.   # code可加性的loss

    def para_load(self, hpara_name):
        hpara_path = hpara_name + ".hyper"

        with open(hpara_path, "rb") as f:
            load_para_dict = pickle.load(f)
            assert len(load_para_dict) == self.param_num

            # load params to the class
            for i in load_para_dict:
                self.__dict__[i] = load_para_dict[i]
            print("load hyper parameters from %s succ: %s" % (hpara_path, str(load_para_dict)))
    
    def para_save(self, hpara_name):
        save_para_dict = {}
        for i in self.hyper_para_list:
            save_para_dict[i] = self.__dict__[i]
        assert len(save_para_dict) == self.param_num

        # save para dict
        hpara_path = hpara_name + ".hyper"
        with open(hpara_path, "wb") as f:
            pickle.dump(save_para_dict, f)
            print("save hyperparameter to %s succ" % hpara_path)

    def _build_encoder(self, inputs):
        # k = tf.constant([1, 1, 1], dtype = tf.float32)
        # k = tf.reshape(k, [int(k.shape[0], 1, 1)], name = "kernel")
        # input_conv1d_1 = tf.reshape(inputs, [-1, int(inputs.shape[0]), 1], name = "inputs_pre_conv1d_1")
        # tf.nn.conv1d(input_conv1d_1, k, 'VALID')
        pooling_input = tf.reshape(inputs, [1, -1, 1])
        pooling_input2 = tf.layers.max_pooling1d(inputs = pooling_input, pool_size = 5, strides = 3, padding = "valid", name = "pooling_1_enc")
        pooling_input2 = tf.layers.max_pooling1d(inputs = pooling_input2, pool_size = 5, strides = 3, padding = "valid", name = "pooling_2_enc")
        l1 = tf.layers.dense(inputs = pooling_input2, units = 512, activation = tf.nn.relu, reuse = tf.AUTO_REUSE,  name = "l1_enc")
        l2 = tf.layers.dense(inputs = l1, units = 256, activation = tf.nn.relu, reuse = tf.AUTO_REUSE, name = "l2_enc")
        l3 = tf.layers.dense(inputs = l2, units = 128, activation = tf.nn.relu, reuse = tf.AUTO_REUSE,  name = "l3_enc")
        encoder_output = tf.layers.dense(inputs = l3, units = 32, activation = tf.nn.relu, reuse = tf.AUTO_REUSE, name = "output_encoder")
        return encoder_output

    def _build_decoder(self, inputs):
        l1 = tf.layers.dense(inputs = inputs, units = 32, activation = tf.nn.relu, reuse = tf.AUTO_REUSE, name = "l1_dec")
        l2 = tf.layers.dense(inputs = l1, units = 128, activation = tf.nn.relu, reuse = tf.AUTO_REUSE, name = "l2_dec")
        l3 = tf.layers.dense(inputs = l2, units = 256, activation = tf.nn.relu, reuse = tf.AUTO_REUSE, name = "l3_dec")
        l4 = tf.layers.dense(inputs = l3, units = 512, activation = tf.nn.relu, reuse = tf.AUTO_REUSE, name = "l4_dec")
        decoder_output = tf.layers.dense(inputs = l4, units = self.theta_num, activation = None, reuse = tf.AUTO_REUSE, name = "output_decoder")
        return decoder_output

    def build_network(self):
        # inputs: K theta.shape = (None, K, theta_num), 2个动态维度
        self.input_theta_ph = tf.placeholder(tf.float32, shape = [None, None, self.theta_num], name = "theta_ph")
        # inputs: weight.shape = (None, 1, K) 2个动态维度
        self.weight_ph = tf.placeholder(tf.float32, shape = [None, 1, None], name = "weight_ph")
        # inputs: target_theta = (None, 1, theta_num) 1 个动态维度
        self.theta_target_ph = tf.placeholder(tf.float32, shape = [None, 1, self.theta_num], name = "theta_target_ph")

        # encoder: shared variables
        with tf.variable_scope("encoder"):
            self.encoder_output = self._build_encoder(self.input_theta_ph)
            self.encoder_output_target = self._build_encoder(self.theta_target_ph)
            
        # decoder
        with tf.variable_scope("decoder"):
            self.decoder_output_utilize = self._build_decoder(tf.matmul(self.weight_ph, self.encoder_output))
            self.decoder_output = self._build_decoder(self.encoder_output)
            self.decoder_output_target = self._build_decoder(self.encoder_output_target)
        
        # define loss
        # loss1: 插值controller参数的恢复loss
        self.loss1 = self.loss1_weight * tf.reduce_mean(tf.squared_difference(self.decoder_output, self.input_theta_ph))
        tf.reshape(self.loss1, (1, ))
        
        # loss2: target controller的恢复loss
        self.loss2 = self.loss2_weight * tf.reduce_mean(tf.squared_difference(self.decoder_output_target, self.theta_target_ph))
        tf.reshape(self.loss2, (1, ))

        # loss3: 插值controller参数的code加权和 = target controller 的code
        self.loss3 = self.loss3_weight * tf.reduce_mean(tf.squared_difference(tf.matmul(self.weight_ph , self.encoder_output), self.encoder_output_target))
        tf.reshape(self.loss3, (1, ))

        # self.loss = self.loss1
        self.loss = self.loss1 + self.loss2 + self.loss3
        opt = tf.train.AdamOptimizer(self.lr)
        grad = opt.compute_gradients(self.loss)
        clipped_grad = [(tf.clip_by_value(grad_, -1, 1), var)  for grad_, var in grad]
        self.train_op = opt.apply_gradients(clipped_grad)

        # get weight
        self.net_weight_dict = {}
        for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            self.net_weight_dict[i.name] = tf.get_default_graph().get_tensor_by_name(i.name)
        
        # session
        self.sess = tf.Session()
        self.log_saver = tf.summary.FileWriter("logs/autoencoder_logs", self.sess.graph)
        self.train_saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def load(self, name, load_conf = True):
        # load weight
        self.train_saver.restore(self.sess, name)
        print("[log] load autoencder from %s" % name)

        # load hyper paras
        if load_conf == True:
            self.para_load(name)
        else:
            print("didn't load conf ")

        # print weight to display
        for i in self.net_weight_dict:
            print("%s : %.2f" % (i, np.linalg.norm(self.sess.run(self.net_weight_dict[i]))))

    def save(self, name):
        # save model
        self.train_saver.save(self.sess, name)
        print("model saved to %s" % name)

        # save hyper parameters
        self.para_save(name)

        # save weight
        variable_dict = {}
        for i in self.net_weight_dict:
            variable_dict[i] = self.sess.run(self.net_weight_dict[i])
        with open(name+".weight", "wb") as f:
            pickle.dump(variable_dict, f)
        print("model weight saved to %s" % (name + ".weight"))
        
    def train_one_step(self, theta_train, theta_target, weights):
        self.total_iter += 1
        self.iter += 1
        # 测试一下shape
        assert theta_train.shape[0] == theta_target.shape[0] == weights.shape[0]
        assert theta_target.shape[1] == 1 and weights.shape[1] == 1
        assert theta_target.shape[2] == theta_train.shape[2] == self.theta_num

        # 输入
        feed_dict = {}
        feed_dict[self.input_theta_ph] = theta_train
        feed_dict[self.theta_target_ph] = theta_target
        feed_dict[self.weight_ph] = weights

        _, loss1, loss2, loss3, loss, decoder_output, inputs = self.sess.run([self.train_op, self.loss1, self.loss2, self.loss3, self.loss,
         self.decoder_output, self.input_theta_ph], feed_dict=feed_dict)
        assert decoder_output.shape == inputs.shape
        # print()
        # print(np.linalg.norm(decoder_output))
        # print(np.linalg.norm(inputs))
        # print(loss)
        # assert 0 == 1
        diff = decoder_output - inputs
        summary = tf.Summary()
        summary.value.add(tag="io_diff/mean", simple_value = float(np.mean(diff)))
        summary.value.add(tag="io_diff/std", simple_value = float(np.std(diff)))
        summary.value.add(tag="Loss/loss1_插值controller的恢复", simple_value = float(loss1))
        summary.value.add(tag="Loss/loss2_target_controller的恢复", simple_value = float(loss2))
        summary.value.add(tag="Loss/loss3_code空间的线性组合关系", simple_value = float(loss3))
        summary.value.add(tag="Loss_normalized/loss1(归一化)", simple_value = float(loss1) / self.loss1_weight)
        summary.value.add(tag="Loss_normalized/loss2(归一化)", simple_value = float(loss2) / self.loss2_weight)
        summary.value.add(tag="Loss_normalized/loss3(归一化)", simple_value = float(loss3) / self.loss3_weight)
        summary.value.add(tag="Loss/sum_loss", simple_value = float(loss))

        # summary.value.add(tag="Rewards/Episode_Length", simple_value = float(step))
        self.log_saver.add_summary(summary, self.total_iter)
        self.log_saver.flush()
        print("\repoch %d, iter %d, loss:%.6f, lr = %.6f" % (self.epoch, self.iter, loss, self.lr), end = '')
        
        return loss

    def forward(self, weight, theta_train):
        # 获取最终的插值结果
        assert weight.shape == (theta_train.shape[0], 1, theta_train.shape[1])
        assert theta_train.shape[2] == self.theta_num
        forward_result = self.sess.run(self.decoder_output_utilize, feed_dict = {
                    self.input_theta_ph : theta_train,
                    self.weight_ph : weight
                })
        # print(forward_result)
        # print(forward_result.shape)
        assert forward_result.shape == (weight.shape[0], 1, self.theta_num)
        return forward_result
        
    def train_one_epoch(self, conf_lst, theta_mat, K, batch_size):
        self.iter = 0 
        self.epoch += 1
        data_num = len(conf_lst)
        loss_lst = []
        # 遍历一次数据集
        for _ in range(0, data_num + batch_size, batch_size):
            # 获取数据
            train_data, target_data, weight_data, pt_id = sample(theta_mat, conf_lst, K, batch_size)
            # 进行训练
            loss = self.train_one_step(train_data, target_data, weight_data)
            loss_lst.append(loss)
        
        # 100个epoch保存一次
        # 10个epoch显示一次
        save_interval = 30
        print_interval = 10
        avg_loss = np.mean(loss_lst[-save_interval:])
        if self.epoch % print_interval == 0:
            print(", avg_loss %.6f" % avg_loss)

        if self.epoch % save_interval ==0:
            save_path = os.path.join(project_dir, "data0824/autoencoder_model/epoch%d-loss%.6f" % (self.epoch, avg_loss))
            self.save(save_path)

        if self.epoch % self.lr_decay_epoches == 0 and self.iter !=0:
            self.lr = max(self.lr * self.lr_decay, self.lr_min)

def train_mimic():
    epochs = 5000
    K = 5
    batch_size = 8
    lr = 1e-3
    lr_decay = 0.9
    lr_min = 1e-6
    lr_decay_epoches = 10

    log_path = "logs/autoencoder_logs"
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
        
    if os.path.exists(log_path) == False:
        os.makedirs(log_path)
    
    # 从参数中读取数据信息文件(info.json，用来描述每个controller和对应skeleton的情况)
    parser = argparse.ArgumentParser("autoencoder arg parser")
    parser.add_argument("--data_info_path", type = str, default=default_info_file_path, help="read a info.json file for the skeleton-configuration pairs")
    arg = parser.parse_args()
    
    # 读取所有controller weight
    weight_paths, skeleton_paths = read_train_data_info(arg.data_info_path)
    theta_mat = read_batch_weight(weight_paths, weight_name_info, weight_shape_info)

    # 读取所有skeleton
    skeleton_lst = []
    for i in skeleton_paths:
        skeleton_lst.append(load_single_skeleton(i))

    # create and train autoencoder
    theta_num = theta_mat.shape[1]
    net = autoencoder(theta_num, lr, lr_decay, lr_decay_epoches, lr_min)

    # load model
    # sa.load("/home/xudong/Projects/controller-space/data/autoencoder_model/epoch1470-loss0.002821", load_conf= True)
    for i in range(epochs):
        net.train_one_epoch(skeleton_lst, theta_mat, K, batch_size)

def test_mimic():
    parser = argparse.ArgumentParser("autoencoder arg parser")
    parser.add_argument("--data_info_path", type = str, default=default_info_file_path, help="read a info.json file for the skeleton-configuration pairs")
    arg = parser.parse_args()
    
    weight_paths, skeleton_paths = read_train_data_info(arg.data_info_path)
    weight_mat = read_batch_weight(weight_paths, weight_name_info, weight_shape_info)

    # read skeleton
    skeleton_lst = []
    for i in skeleton_paths:
        skeleton_lst.append(load_single_skeleton(i))

    # begin to train
    # initialize the autoencoder
    para_num = sum([i[0] * i[1] if 2 == len(i) else i[0] for i in weight_shape_info])
    # sa = autoencoder(theta_num = para_num)

    # sample 
    train_data_total, target_data_total, weight_data_total, center_pt_id = sample(weight_mat, skeleton_lst, 5, 8)
    # print(weight_data_total)

if __name__ == "__main__":
    # 测试lr参数 = 0.01, 0.001, 0.0001
    # lr_decay = 0.95, 0.7, 0.5
    # batch_size = 64, 32, 16, 8
    # K = 3, 5, 7, 10
    train_mimic()
    # test_mimic()
    # sa = autoencoder()
    # test()
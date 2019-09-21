import numpy as np
import copy as copy
import tensorflow as tf
import pickle
import time, datetime
import os

from util.logger import Logger
import util.mpi_util as MPIUtil
import util.math_util as MathUtil
import learning.rl_util as RLUtil
import learning.tf_util as TFUtil
from env.env import Env
from learning.path_xudong import *
from learning.solvers.mpi_solver import MPISolver
import learning.nets.net_builder as NetBuilder

from learning.replay_buffer_xudong import ReplayBuffer_xudong
from learning.rl_agent_xudong import Agent_xudong

'''
    PPOAgent_xudong, inherited from agent_xudong
'''
para_get = lambda key_name, default, json: default if (key_name not in json) else json[key_name]      

class PPOAgent_xudong(Agent_xudong):
    NAME = "PPO"
    
    # actor
    ACTOR_NET = "ActorNet"
    ACTOR_STEP_SIZE = "ActorStepsize"
    ACTOR_MOMENTUM = "ActorMomentum"
    ACTOR_WEIGHT_DECAY = "ActorWeightDecay"
    
    # critic 
    CRITIC_NET = "CriticNet"
    CRITIC_STEP_SIZE = "CriticStepsize"
    CRITIC_MOMENTUM = "CriticMomentum"
    CRITIC_WEIGHT_DECAY = "CriticWeightDecay"

    # hypers
    DISCOUNT = "Discount"
    EPOCHS_KEY = "Epochs"
    BATCHSIZE = "BatchSize"
    MINI_BATCH_SIZE = "MiniBatchSize"
    REPLAYBUFFER_SIZE = "ReplayBufferSize"
    INIT_SAMPLES_KEY = "InitSamples"
    RATIO_CLIP = "RatioClip"
    TD_LAMBDA = "TDLambda"
    TEST_EPISODES = "TestEpisodes"
    ITERS_PER_UPDATE = "ItersPerUpdate"
    OUTPUT_ITERS_KEY = "OutputIters"
    INT_OUTPUT_ITERS_KEY = "IntOutputIters"

    # others
    EXP_ACTION_FLAG = 1 << 0
    def __init__(self, world, id, json_data):
        super().__init__(world, id)
        # load hyper params
        self.load_params(json_data)

        self._build_graph(json_data)
        self._build_replay_buffer(self.replay_buffer_size)
        self._build_saver()
        self.log_saver = tf.summary.FileWriter("logs/controller_logs/board/", self.sess.graph)
        self.logger = Logger()
        return 
    
    def compute_value_term(self, discount):
        self.val_min, self.val_max = self._calc_val_bounds(discount)
        self.val_fail, self.val_succ = self._calc_term_vals(discount)

    def compute_buffer_size(self):
        # replay buffer size >= 2 * (batch_size / procs)
        num_procs = MPIUtil.get_num_procs()
        local_batch_size = int(self.batch_size / num_procs)
        min_replay_size = 2 * local_batch_size
        assert (self.replay_buffer_size > min_replay_size)

        self.replay_buffer_size = np.maximum(min_replay_size, self.replay_buffer_size)

    def _initialize_vars(self):
        self.sess.run(tf.global_variables_initializer())
        return

    def load_params(self, json_data):
        super().load_params(json_data)
        
        # read hyper paras from config
        self.discount = para_get(self.DISCOUNT, 0.95, json_data)
        self.epochs = para_get(self.EPOCHS_KEY, 1, json_data)
        self.batch_size = para_get(self.BATCHSIZE, 1024, json_data)
        self.mini_batch_size = para_get(self.MINI_BATCH_SIZE, 32, json_data)
        self.replay_buffer_size = para_get(self.REPLAYBUFFER_SIZE, 1024000, json_data)
        self.init_samples = para_get(self.INIT_SAMPLES_KEY, 1, json_data)
        self.ratio_clip = para_get(self.RATIO_CLIP, 0.2, json_data)
        self.td_lambda = para_get(self.TD_LAMBDA, 0.95, json_data)
        self.test_episodes = para_get(self.TEST_EPISODES, 32, json_data)
        self.iters_per_update = para_get(self.ITERS_PER_UPDATE, 1, json_data)
        self.output_iters = para_get(self.OUTPUT_ITERS_KEY, 1, json_data)
        self.int_output_iters = para_get(self.INT_OUTPUT_ITERS_KEY, 10, json_data)

        self.compute_buffer_size()
        self.compute_value_term(self.discount)

    def _build_graph(self, json_data):
        with self.sess.as_default(), self.graph.as_default():
            self._build_nets(json_data)         # static graph structure, must be the first initialized
            self._build_losses(json_data)       # losses definition, it points out how to train the agent
            self._build_solvers(json_data)      # employ an uncommon optimizer, so called "solver"
            self._initialize_vars()             # intializer vars
            self._build_saver()                 # build saver

    def _build_nets(self, json_data):
        '''
            actor-critic style, build 2 nets
        '''
        assert self.ACTOR_NET in json_data
        assert self.CRITIC_NET in json_data

        actor_net_name = json_data[self.ACTOR_NET]
        critic_net_name = json_data[self.CRITIC_NET]

        # actor net ph
        with tf.variable_scope("input"):
            self.state_ph = tf.placeholder(tf.float32, shape = [None, self.state_size], name = "state_ph")
            self.given_action_ph = tf.placeholder(tf.float32, shape = [None, self.action_size], name = "given_action_ph")
            self.old_prob_log_ph = tf.placeholder(tf.float32, shape = [None, 1], name = "old_action_prob_log_ph")
            self.target_value_ph = tf.placeholder(tf.float32, shape = [None, 1], name = "target_value_ph")
            self.adv_value_ph = tf.placeholder(tf.float32, shape = [None, 1], name = "advantage_value_ph")
        
        # build actor: action in [-1, 1]
        with tf.variable_scope("actor"):
            h = NetBuilder.build_net(actor_net_name, self.state_ph, reuse = False)
            self.action_mean_tf = tf.layers.dense(inputs = h, units = self.action_size, activation = tf.nn.tanh)
            self.action_std_tf = tf.layers.dense(inputs = h, units = self.action_size, activation = tf.nn.softplus) / 10

            # 我要构造的应该是一个
            dist = tf.contrib.distributions.Normal(loc = self.action_mean_tf, scale = self.action_std_tf)
            self.sample_action_tf = dist.sample()
            
            self.sample_action_prob_log_tf = TFUtil.calc_logp_gaussian(x_tf = self.sample_action_tf,
                mean_tf = self.action_mean_tf, std_tf = self.action_std_tf)

            self.given_action_prob_log_tf = TFUtil.calc_logp_gaussian(x_tf = self.given_action_ph,
                mean_tf = self.action_mean_tf, std_tf = self.action_std_tf)
            print("[ppo agent] actor net created")

        # build ciritc: value in [0, inf]
        with tf.variable_scope("critic"):
            c = NetBuilder.build_net(critic_net_name, self.state_ph, reuse = False)
            self.critic_value_tf = tf.layers.dense(inputs = c, units = 1, activation = tf.nn.relu)
            print("[ppo agent] critic net created")

    def weight_decay_loss(self, scope):
        
        with self.sess.as_default(), self.graph.as_default():
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = scope)
            assert len(vars) > 0
        
        vars_no_bias = [v for v in vars if 'bias' not in v.name]
        loss = tf.add_n([tf.nn.l2_loss(v) for v in vars_no_bias])
        return loss

    def action_exceed_loss(self, action):
        min_part = tf.maximum(action - self.action_lower_bound, 0)
        max_part = tf.maximum(action - self.action_upper_bound, 0)
        loss = tf.reduce_sum(tf.square(min_part) + tf.square(max_part))
        return loss

    def _build_losses(self, json_data):
        actor_weight_decay = para_get(self.ACTOR_WEIGHT_DECAY, 0, json_data)
        critic_weight_decay = para_get(self.CRITIC_WEIGHT_DECAY, 0, json_data)

        # critic loss = 0.5*(target_val - critic_output) ^2 + weight_decay * weight_sum
        self.critic_loss_tf = tf.reduce_mean(tf.squared_difference(self.target_value_ph, self.critic_value_tf) * 0.5)
        self.critic_loss_tf += critic_weight_decay * self.weight_decay_loss("critic")
    
        # actor loss = PPO classic
        # 这里rator = 对于给定的aciton, 用当前的mean, std计算prob log，然后除以old_prob(也是一个外部的输入)
        self.ratio_tf = tf.exp(self.given_action_prob_log_tf - self.old_prob_log_ph)
        self.actor_loss_1_tf = self.adv_value_ph * self.ratio_tf
        self.actor_loss_2_tf = self.adv_value_ph * tf.clip_by_value(self.ratio_tf, clip_value_min = 1.0 - self.ratio_clip, clip_value_max = 1.0 + self.ratio_clip)
        self.actor_loss_3_tf = self.action_exceed_loss(self.given_action_ph)    # actor exceed bound penalty
        self.actor_loss_4_tf = actor_weight_decay * self.weight_decay_loss("actor")    # weight decay
        self.actor_loss_tf = -tf.reduce_mean(tf.minimum(self.actor_loss_1_tf, self.actor_loss_2_tf)) + self.actor_loss_3_tf + self.actor_loss_4_tf

    def _build_solvers(self, json_data):
        actor_stepsize = para_get(self.ACTOR_STEP_SIZE, 1e-5, json_data)
        actor_momentum = para_get(self.ACTOR_MOMENTUM, 0.9, json_data)
        critic_stepsize = para_get(self.CRITIC_STEP_SIZE, 0.01, json_data)
        critic_momentum = para_get(self.CRITIC_MOMENTUM, 0.9, json_data)

        # build critic sovler
        critic_vars = self.tf_vars("critic")
        critic_opt = tf.train.MomentumOptimizer(learning_rate = critic_stepsize, momentum = critic_momentum)
        self.critic_grad_tf = tf.gradients(self.critic_loss_tf, critic_vars)
        self.critic_solver = MPISolver(self.sess, critic_opt, critic_vars)

        # build actor sovler
        self._actor_stepsize_tf = tf.get_variable(dtype=tf.float32, name='actor_stepsize', initializer=actor_stepsize, trainable=False)
        self._actor_stepsize_ph = tf.get_variable(dtype=tf.float32, name='actor_stepsize_ph', shape=[])
        self._actor_stepsize_update_op = self._actor_stepsize_tf.assign(self._actor_stepsize_ph)

        actor_vars = self.tf_vars('actor')
        actor_opt = tf.train.MomentumOptimizer(learning_rate=self._actor_stepsize_tf, momentum=actor_momentum)
        self.actor_grad_tf = tf.gradients(self.actor_loss_tf, actor_vars)
        self.actor_solver = MPISolver(self.sess, actor_opt, actor_vars)
        
        return 

    def _build_replay_buffer(self, buffer_size):
        self.path = Path()
        num_procs = MPIUtil.get_num_procs()
        buffer_size = int(buffer_size / num_procs)
        self.replay_buffer = ReplayBuffer_xudong(buffer_size=buffer_size)
        self.replay_buffer.add_filter_key(self.EXP_ACTION_FLAG)
        self.replay_buffer_initialized = False
        return
    
    def decide_action(self, s):
        assert np.isfinite(s).all() == True
        s = s.reshape(-1, self.state_size)
        assert s.shape[1] == self.state_size
        num = s.shape[0]

        fdict = { self.state_ph : s }
        with self.sess.as_default(), self.graph.as_default():
            mean, std = self.sess.run([self.action_mean_tf, self.action_std_tf], feed_dict = fdict)
            action, prob_log = self.sess.run([self.sample_action_tf,  self.sample_action_prob_log_tf],
                        feed_dict=fdict)

        # print("action mean = {}".format(mean))
        # print("action std = {}".format(std))
        # print("action = {}".format(action))
        # print("action prob_log = {}".format(prob_log))

        action = action.reshape(-1, self.action_size)
        prob_log = prob_log.reshape(-1, 1)

        assert action.shape == (num, self.action_size)
        assert prob_log.shape == (num, 1)


        return action, prob_log

    def _compute_batch_vals(self, start_idx, end_idx):
        states = self.replay_buffer.get_all("states")[start_idx : end_idx]  # 获取一大段state
        idx = np.array(list(range(start_idx, end_idx))) # 对于这些state的索引而言

        # 对于不同的结尾状态，需要更改其reward
        is_end =self.replay_buffer.get_path_end(idx)    # 对于idx这个array中，给出每个状态的end的序号是多少
        is_fail = self.replay_buffer.check_terminal_flag(idx, Env.Terminate.Fail)   # 是不是fail?
        is_succ = self.replay_buffer.check_terminal_flag(idx, Env.Terminate.Succ)   # 是不是succ?
        
        is_fail = np.logical_and(is_end, is_fail)   # end下的fail才是真正的fail
        is_succ = np.logical_and(is_end, is_succ)   # end下的succ才是真正的succ

        assert states.shape[1] == self.state_size
        num = states.shape[0]
        fdict = {self.state_ph : states}
        vals = self.sess.run(self.critic_value_tf, feed_dict=fdict) # 利用网络去估计这段state的value
        vals = vals.reshape(num)
        assert vals.shape == (num, )

        vals[is_fail] = self.val_fail   # 凡是这段state中判定为fail的，都要特殊将value赋值为 value_fail
        vals[is_succ] = self.val_succ   # 凡是end succ的，要赋值为val_succ
        
        return vals

    def _compute_batch_new_vals(self, start_idx, end_idx, val_buffer):
        # compute return from reward
        assert len(val_buffer.shape) == 1
        rewards = self.replay_buffer.get_all("rewards")[start_idx: end_idx]

        if self.discount is 0 :
            new_vals = rewards.copy()
        else:
            new_vals = np.zeros_like(val_buffer)

            curr_idx = start_idx
            while curr_idx < end_idx:
                idx0 = curr_idx - start_idx
                idx1 = self.replay_buffer.get_path_end(curr_idx) - start_idx
                r = rewards[idx0: idx1]
                v = val_buffer[idx0 : (idx1+1)]
                new_vals[idx0:idx1] = RLUtil.compute_return(r, self.discount, self.td_lambda, v)   # GAE
                curr_idx = idx1 + start_idx + 1
        return new_vals

    def _update_critic(self, state, target_vals):
        feed = {
            self.state_ph: state,
            self.target_value_ph: target_vals
        }

        loss, grads = self.sess.run([self.critic_loss_tf, self.critic_grad_tf], feed)
        self.critic_solver.update(grads)
        return loss

    def _update_actor(self, state, action, old_prob_log, adv):

        feed = {
            self.state_ph : state,
            self.adv_value_ph : adv,
            self.given_action_ph : action,
            self.old_prob_log_ph : old_prob_log,
        }
        loss, grads = self.sess.run([self.actor_loss_tf, self.actor_grad_tf], feed)
        self.actor_solver.update(grads)
        return loss

    def _train_step(self):
        ''' for each substep '''

        # get the buffer info, st to ed
        start_idx = self.replay_buffer.buffer_tail
        end_idx = self.replay_buffer.buffer_head
        assert (start_idx == 0)
        assert (self.replay_buffer.get_current_size() <= self.replay_buffer.buffer_size)
        assert (start_idx < end_idx)

        # buffer idx array
        idx = np.array(list(range(start_idx, end_idx)))
        end_mask = self.replay_buffer.is_path_end(idx)
        end_mask = np.logical_not(end_mask)

        # compute returns for each state (using GAE)
        vals = self._compute_batch_vals(start_idx, end_idx)
        new_vals = self._compute_batch_new_vals(start_idx, end_idx, vals)

        # idx是一个buffer长度的array, end_mask是一个0 1 序列
        # 结果会选中那些true的，丢掉那些false的; 
        # 所以valid_idx是那些不是end path的state id
        valid_idx = idx[end_mask]
        # exp_idx是replay buffer中, exp_action_flag对应的buffer的idx
        exp_idx = self.replay_buffer.get_idx_filtered(self.EXP_ACTION_FLAG).copy()
        num_valid_idx = valid_idx.shape[0]  # 有效idx有多少个？
        num_exp_idx = exp_idx.shape[0]      # exp idx有多少个?
        # 吧exp_idx和一个[0-exp_idx]作为2列，拼成一个matrix
        exp_idx = np.column_stack([exp_idx, np.array(list(range(0, num_exp_idx)), dtype=np.int32)])
        
        # valid_idx = idx[end_mask]
        # num_valid_idx = valid_idx.shape[0]
        # 本地采样数
        local_sample_count = valid_idx.size
        global_sample_count = int(MPIUtil.reduce_sum(local_sample_count))   # 所有进程的采样数相加;得到总采样数 
        mini_batches = int(np.ceil(global_sample_count / self.mini_batch_size)) # 然后得到mini batch 个数
        # 从中我们就知道，采样一定需要共享。一起进行训练。
        
        # 优势函数的计算: 只把所有exp_idx的拿出来进行训练，这是为什么?
        adv = new_vals[valid_idx] - vals[valid_idx] # adv = 
        new_vals = np.clip(new_vals, self.val_min, self.val_max)    # 以前的new_vals也都更新了; 存在一个max

        # adv mean, adv std: 优势函数的mean & std
        # adv 也转化到了方差=1和mean=0的分布。
        # 暂时去掉advantage的正则化
        adv_mean = np.mean(adv)
        adv_std = np.std(adv)
        # adv = (adv - adv_mean) / (adv_std + adv_eps)
        # adv = np.clip(adv, -self.norm_adv_clip, self.norm_adv_clip)

        critic_loss = 0
        actor_loss = 0
        # 每次采样完开始训练的时候, 是不是一起训练?
        for e in range(self.epochs):
            print("[ppo agent train step] epoch %d" % e)
            np.random.shuffle(valid_idx)    # shuffle 很重要

            for b in range(mini_batches):
                # 获取当前mini batch的idx
                batch_idx_begin = b * self.mini_batch_size
                batch_idx_end = batch_idx_begin + self.mini_batch_size

                critic_batch = np.array(range(batch_idx_begin, batch_idx_end), dtype=np.int32)
                actor_batch = critic_batch.copy()
                critic_batch = np.mod(critic_batch, num_valid_idx)
                actor_batch = np.mod(actor_batch, num_exp_idx)
                shuffle_actor = (actor_batch[-1] < actor_batch[0]) or (actor_batch[-1] == num_exp_idx - 1)

                critic_batch = valid_idx[critic_batch]
                actor_batch = exp_idx[actor_batch]
                critic_batch_vals = new_vals[critic_batch]
                actor_batch_adv = adv[actor_batch]  # adv = gae_val - val

                # update critic network
                critic_s = self.replay_buffer.get_all("states")
                curr_critic_loss = self._update_critic(critic_s, critic_batch_vals)

                # update actor network
                actor_s = self.replay_buffer.get("states", actor_batch[:,0])  # 必须得有goal,不然怎么mimic?
                actor_a = self.replay_buffer.get("actions", actor_batch[:,0])
                actor_prob = self.replay_buffer.get("logps", actor_batch[:,0])

                curr_actor_loss = self._update_actor(actor_s, actor_a, actor_prob, actor_batch_adv)
                # print("[train log] sub epoch %d loss = %.3f" % (b, curr_actor_loss))
                assert np.isfinite(curr_actor_loss).all() == True

                critic_loss += curr_critic_loss
                actor_loss += np.abs(curr_actor_loss)

                if (shuffle_actor):
                    np.random.shuffle(exp_idx)

        total_batches = mini_batches * self.epochs
        critic_loss /= total_batches
        actor_loss /= total_batches

        critic_loss = MPIUtil.reduce_avg(critic_loss)
        actor_loss = MPIUtil.reduce_avg(actor_loss)

        critic_stepsize = self.critic_solver.get_stepsize()
        actor_stepsize = self.actor_solver.get_stepsize()
        # 如果clip的频率过大的话，就要对步长进行调整
        # clip频率大了说明他一步走的略大，所以应该调整。
        # 这个策略可以暂时放弃
        # actor_stepsize = self.update_actor_stepsize(actor_clip_frac)
        
        self.logger.log_tabular('Critic_Loss', critic_loss)
        self.logger.log_tabular('Critic_Stepsize', critic_stepsize)
        self.logger.log_tabular('Actor_Loss', actor_loss) 
        self.logger.log_tabular('Actor_Stepsize', actor_stepsize)
        self.logger.log_tabular('Adv_Mean', adv_mean)
        self.logger.log_tabular('Adv_Std', adv_std)

        self.replay_buffer.clear()

        return
    def end_episode(self):
        '''
            store path(state-action-reward) to the replay buffer
        '''
        if(self.path.pathlength() > 0):
            self._end_path()    # 把最后的s-r-g都放path里面

            if (self._mode is self.Mode.TRAIN or self._mode is self.Mode.TRAIN_END):
                if (self.path.pathlength() > 0):
                    # train model : save path into the replay buffer
                    self._store_path(self.path)
            elif (self._mode == self.Mode.TEST):
                self._update_test_return(self.path)
            else:
                assert False, Logger.print("Unsupported agent mode" + str(self._mode))
            self._update_mode()
        return 

    def _update_mode(self):
        if (self._mode == self.Mode.TRAIN):
            self._update_mode_train()
        elif (self._mode == self.Mode.TRAIN_END):
            self._update_mode_train_end()
        elif (self._mode == self.Mode.TEST):
            self._update_mode_test()
        else:
            assert False, Logger.print("Unsupported RL agent mode" + str(self._mode))
        return

    def _update_mode_train(self):
        return

    def _update_mode_train_end(self):
        self._init_mode_test()
        return

    def _update_mode_test(self):
        if (self.test_episode_count * MPIUtil.get_num_procs() >= self.test_episodes):
            global_return = MPIUtil.reduce_sum(self.test_return)
            global_count = MPIUtil.reduce_sum(self.test_episode_count)
            avg_return = global_return / global_count
            self.avg_test_return = avg_return

            self._init_mode_train()
        return

    def _update_test_return(self, path):
        path_reward = path.calc_return()
        self.test_return = path_reward
        self.test_episode_count += 1

    def _init_mode_train(self):
        self._mode = self.Mode.TRAIN
        self.world.env.set_mode(self._mode)
        return

    def _init_mode_train_end(self):
        self._mode = self.Mode.TRAIN_END
        return

    def _init_mode_test(self):
        self._mode = self.Mode.TEST
        self.test_return = 0.0
        self.test_episode_count = 0
        self.world.env.set_mode(self._mode)
        return
    # def _valid_train_step(self):
    #     # 只有当批量达到batch size的时候才行
    #     samples = self.replay_buffer.get_current_size()
    #     exp_samples = self.replay_buffer.count_filtered(self.EXP_ACTION_FLAG)
    #     global_sample_count = int(MPIUtil.reduce_sum(samples))
    #     global_exp_min = int(MPIUtil.reduce_min(exp_samples))
    #     # print("[valid train step] current %d samples > batchsize = %d" % (global_sample_count, self.batch_size))
    #     return (global_sample_count > self.batch_size) and (global_exp_min > 0)

    def save_model(self, out_path):
        # save model 
        with self.sess.as_default(), self.graph.as_default():
            try:
                save_path = self.saver.save(self.sess, out_path, write_meta_graph=False, write_state=False)
            except:
                Logger.print("Failed to save model to: " + save_path)

        # save weight
        weight_lst = self.tf_vars("actor/")
        weight_dict = {}
        name_lst = []
        size = 0
        for i in weight_lst:
            name_lst.append(i.name)
            weight_dict[i.name] = self.sess.run(i)
            # print((i.name, weight_dict[i.name].shape))
            size += weight_dict[i.name].size
        # print("sum size = %d" % size)
        weight_save_path = save_path + ".weight"
        with open(weight_save_path, "wb") as f:
            pickle.dump(weight_dict, f)
        
        Logger.print('[ppo_agent_xudong] Model saved to: ' + save_path)
        Logger.print('[ppo_agent_xudong] Model weight saved to : ' + weight_save_path)
        return

    def load_model(self, in_path):
        with self.sess.as_default(), self.graph.as_default():
            self.saver.restore(self.sess, in_path)
            Logger.print('[ppo_agent_xudong] Model loaded from: ' + in_path)
        return

    def _end_path(self):
        s = self._record_state()
        g = self._record_goal()
        r = self._record_reward()
        
        self.path.rewards.append(r)
        print("[ppo_agent_xudong] end path, total r = {}".format(sum(self.path.rewards)))
        self.path.states.append(s)
        assert np.isfinite(s).all() == True # 在end of path的时候，state突然崩了。
        # 其实我还有点好奇: state为什么是275呢?
        self.path.terminate = self.world.env.check_terminate(self.id)

        return

    def _store_path(self, path):
        path_id = self.replay_buffer.store(path)
        valid_path = path_id != MathUtil.INVALID_IDX

        if valid_path:
            self.train_return = path.calc_return()

        return path_id

    def _record_state(self):
        # 返回当前状态
        s = self.world.env.record_state(self.id)
        return s

    def _record_goal(self):
        g = self.world.env.record_goal(self.id)
        return g

    def _record_reward(self):
        r = self.world.env.calc_reward(self.id)
        return r
    
    def update(self, timestep):
        # 更新当前agent: agent自己决定是否需要设置为新的action, 并且自己决定是否要训练。
        if True == self.need_new_action():
            # 自己决定是否需要设置新的action(通过state输入网络并且前向传播)
            self._update_new_action()

        # 看是否训练: 处在训练的模式时则训练
        if (self._mode == self.Mode.TRAIN):
            self._train()
            self.world.env.set_sample_count(self._total_sample_count)
            
    def reset(self):
        super().reset()
        self.path.clear()

    def _update_new_action(self):
        s = self._record_state()
        flags = self._record_flags()
        s = s.reshape([self.state_size,])
        assert s.shape == (self.state_size, )
        # 因为这只是一个update_new_action, 而不是真正的apply action
        # 所以接下来拿到的reward，是上一步所采取的action的reward;
        # 所以如果是第一步，则reward不应该入库; 
        # 如果不是第一步(如下代码), reward应该入库
        if False == self._is_first_step():
            r = self._record_reward()
            assert type(r) == float
            self.path.rewards.append(r)
        assert np.isfinite(s).all() == True

        # decide action
        a, prob_log = self.decide_action(s = s)
        a, prob_log = a.reshape(a.shape[1]), prob_log.reshape(prob_log.shape[1])
        assert a.shape == (self.action_size, )
        assert prob_log.shape == (1, )

        # verify and set action
        assert np.isfinite(a).all() == True
        self._apply_action(a)

        self.path.states.append(s)
        self.path.actions.append(a)
        self.path.logps.append(prob_log)
        self.path.flags.append(flags)

    def _is_first_step(self):
        return len(self.path.states) == 0

    def _train(self):
        samples = self.replay_buffer.total_count

        self._total_sample_count = int(MPIUtil.reduce_sum(samples))
        end_training = False   # 不知道是干什么的

        if(self.replay_buffer_initialized):
            if (self._valid_train_step()):
                
                prev_iter = self.iter
                iters =self._get_iters_per_update() # proc_num * iters_per_update

                avg_train_return = MPIUtil.reduce_avg(self.train_return)
                self.avg_train_return = avg_train_return

                for i in range(iters):
                    curr_iter = self.iter

                    # how much time have this program run ?(hour)
                    wall_time = time.time() - self.start_time
                    wall_time /= 60 * 60

                    self.logger.log_tabular("Samples", self._total_sample_count)
                    self.logger.log_tabular("Train_Return", avg_train_return)
                    self.logger.log_tabular("Test_Return", self.avg_test_return)

                    self._update_iter(self.iter + 1)

                    # train step (vital)
                    self._train_step()

                    Logger.print("Agent " + str(self.id))

                    self.logger.print_tabular()
                    Logger.print("")

                    if(curr_iter % self.int_output_iters == 0):
                        self.logger.dump_tabular()

                # judge whether it needs to test
                if (prev_iter // self.int_output_iters != self.iter // self.int_output_iters):
                    end_training = self.enable_testing()
        else:
            print("[ppo_agent_xudong] train called 4")
            Logger.print("Agent " + str(self.id))
            Logger.print("Samples: " + str(self._total_sample_count))
            Logger.print("")

            # 如果即将开始第一次训练，那么这个时候先test一下,看看多少分。
            if (self._total_sample_count >= self.init_samples):
                self.replay_buffer_initialized = True
                end_training = self.enable_testing()
            # else:
                # print("total sample count = {}".format(self._total_sample_count))
        if end_training:
            self._init_mode_train_end()

        return 
    def _valid_train_step(self):
        # 满足一个batch就ok了
        samples = self.replay_buffer.get_current_size()
        global_sample_count = int(MPIUtil.reduce_sum(samples))
        valid =(global_sample_count > self.batch_size)
        if valid == False:
            print("invalid train step, now global sample = {} < batch size {}".format(global_sample_count, self.batch_size)) 
        return valid

    def _update_iter(self, iter):
        if (self.iter % self.output_iters == 0):

            # 只有root proc才有权保存模型
            # 可以设置多个iter保存一次模型, what ever
            output_path = self._get_output_path()
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            self.save_model(output_path)

        self.iter = iter
        return

    def _enable_output(self):
        return MPIUtil.is_root_proc() and self.output_dir != ""

    def _build_saver(self):
        vars = self._get_saver_vars()
        [print(i) for i in vars]
        self.saver = tf.train.Saver(vars, max_to_keep=0)
        return

    def _get_saver_vars(self):
        with self.sess.as_default(), self.graph.as_default():
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            # vars = [v for v in vars if '/' + self.SOLVER_SCOPE + '/' not in v.name]
            #vars = [v for v in vars if '/target/' not in v.name]
            assert len(vars) > 0

        return vars
    
    def _record_flags(self):
        return 1

    def enable_testing(self):
        return self.test_episodes > 0

    def _get_iters_per_update(self):
        return MPIUtil.get_num_procs() * self.iters_per_update

    def _get_output_path(self):
        assert(self.output_dir != '')
        self.train_return
        file_path = self.output_dir + '/agent' + str(self.id) + "_model_" +\
         str(datetime.datetime.now())[:19].replace(" ", "_").replace("-","_").replace(":","_") + \
            str("_%.2f" % self.avg_train_return) + ".ckpt"
        return file_path
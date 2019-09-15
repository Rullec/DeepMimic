import numpy as np
import copy as copy
import tensorflow as tf
import pickle

from util.logger import Logger
import util.mpi_util as MPIUtil
import util.math_util as MathUtil
import learning.rl_util as RLUtil
from env.env import Env
from learning.path import *
from learning.solvers.mpi_solver import MPISolver
import learning.nets.net_builder as NetBuilder

# from learning.replay_buffer import ReplayBuffer
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
    RATIO_CLIP = "RatioClip"
    TD_LAMBDA = "TDLambda"
    TEST_EPISODES = "TestEpisodes"

    def __init__(self, world, id, json_data):
        super().__init__(world, id)
        # load hyper params
        self.load_params(json_data)

        self._build_graph(json_data)
        self._build_replay_buffer(self.replay_buffer_size)
        self.log_saver = tf.summary.FileWriter("logs/controller_logs/board/", self.sess.graph)
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

        
    def load_params(self, json_data):
        super().load_params(json_data)
        
        # read hyper paras from config
        self.discount = para_get(self.DISCOUNT, 0.95, json_data)
        self.epochs = para_get(self.EPOCHS_KEY, 1, json_data)
        self.batch_size = para_get(self.BATCHSIZE, 1024, json_data)
        self.mini_batch_size = para_get(self.MINI_BATCH_SIZE, 32, json_data)
        self.replay_buffer_size = para_get(self.REPLAYBUFFER_SIZE, 1024000, json_data)
        self.ratio_clip = para_get(self.RATIO_CLIP, 0.2, json_data)
        self.td_lambda = para_get(self.TD_LAMBDA, 0.95, json_data)
        self.test_episodes = para_get(self.TEST_EPISODES, 32, json_data)

        self.compute_buffer_size()
        self.compute_value_term(self.discount)

    def _build_graph(self, json_data):
        with self.sess.as_default(), self.graph.as_default():
            self._build_nets(json_data)      # static graph structure, must be the first initialized
            self._build_losses(json_data)    # losses definition, it points out how to train the agent
            self._build_solvers(json_data)   # employ an uncommon optimizer, so called "solver"

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
            self.old_prob_ph = tf.placeholder(tf.float32, shape = [None], name = "old_action_prob_ph")
            self.target_value_ph = tf.placeholder(tf.float32, shape = [None], name = "target_value_ph")
            self.adv_value_ph = tf.placeholder(tf.float32, shape = [None], name = "advantage_value_ph")
        
        # build actor: action in [-1, 1]
        with tf.variable_scope("actor"):
            prob_eps = 1e-6
            h = NetBuilder.build_net(actor_net_name, self.state_ph, reuse = False)
            self.action_mean_tf = tf.layers.dense(inputs = h, units = self.action_size, activation = tf.nn.tanh)
            self.action_std_tf = tf.layers.dense(inputs = h, units = self.action_size, activation = tf.nn.softplus) + prob_eps

            dist = tf.contrib.distributions.Normal(loc = self.action_mean_tf, scale = self.action_std_tf)
            self.sample_action_tf = dist.sample(1)
            self.sample_action_prob_tf = dist.prob(self.sample_action_tf) + prob_eps
            self.given_action_prob_tf = dist.prob(self.given_action_ph) + prob_eps
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
        self.ratio_tf = tf.exp(tf.log(self.sample_action_prob_tf) - tf.log(self.old_prob_ph))
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
        self.replay_buffer_initialized = False
        return
    
    def decide_action(self, s):
        assert np.isfinite(s).all() == True
        assert s.shape == (-1, self.state_size)
        num = s.shape[0]

        fdict = { self.state_ph : s }
        with self.sess.as_default(), self.graph.as_default():
            action, prob = self.sess.run([self.sample_action_tf, self.sample_action_prob_tf], feed_dict=fdict)

        assert action.shape == (num, self.action_size)
        assert prob.shape == (num, 1)
        return action, prob

    def _compute_batch_vals(self, start_idx, end_idx):
        states = self.replay_buffer.get_all("states")[start_idx : end_idx]  # 获取一大段state
        idx = np.array(list(range(start_idx, end_idx))) # 对于这些state的索引而言

        # 对于不同的结尾状态，需要更改其reward
        is_end =self.replay_buffer.get_path_end(idx)    # 对这一段中每一个state都判断: 他是不是end?
        is_fail = self.replay_buffer.check_terminal_flag(idx, Env.Terminate.Fail)   # 是不是fail?
        is_succ = self.replay_buffer.check_terminal_flag(idx, Env.Terminate.Succ)   # 是不是succ?
        
        is_fail = np.logical_and(is_end, is_fail)   # end下的fail才是真正的fail
        is_succ = np.logical_and(is_end, is_succ)   # end下的succ才是真正的succ

        assert states.shape == (-1, self.state_size)
        num = states.shape[0]
        fdict = {self.state_ph : states}
        vals = self.sess.run(self.critic_value_tf, feed_dict=fdict) # 利用网络去估计这段state的value
        assert vals.shape == (num, 1)

        vals[is_fail] = self.val_fail   # 凡是这段state中判定为fail的，都要特殊将value赋值为 value_fail
        vals[is_succ] = self.val_succ   # 凡是end succ的，要赋值为val_succ
        
        return vals

    def _compute_batch_new_vals(self, start_idx, end_idx, val_buffer):
        # compute return from reward
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
                new_vals = RLUtil.compute_return(r, self.discount, self.td_lambda, v)   # GAE
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

    def _update_actor(self, state, action, old_prob, adv):

        feed = {
            self.state_ph : state,
            self.adv_value_ph : adv,
            self.given_action_ph : action,
            self.given_action_prob_tf : old_prob,
        }
        loss, grads = self.sess.run([self.actor_loss_tf, self.actor_grad_tf], feed)
        self.actor_solver.update(grads)
        return loss

    def _train_step(self):
        ''' for each substep '''
        adv_eps = 1e-5

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

        # --------暂时停止exp_idx的使用-------------
        # # idx是一个buffer长度的array, end_mask是一个0 1 序列
        # # 结果会选中那些true的，丢掉那些false的; 
        # # 所以valid_idx是那些不是end path的state id
        # valid_idx = idx[end_mask]
        # # exp_idx是replay buffer中, exp_action_flag对应的buffer的idx
        # exp_idx = self.replay_buffer.get_idx_filtered(self.EXP_ACTION_FLAG).copy()
        # num_valid_idx = valid_idx.shape[0]  # 有效idx有多少个？
        # num_exp_idx = exp_idx.shape[0]      # exp idx有多少个?
        # # 吧exp_idx和一个[0-exp_idx]作为2列，拼成一个matrix
        # exp_idx = np.column_stack([exp_idx, np.array(list(range(0, num_exp_idx)), dtype=np.int32)])
        
        valid_idx = idx[end_mask]
        num_valid_idx = valid_idx.shape[0]
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
                actor_batch = critic_batch.copy()

                critic_batch = valid_idx[critic_batch]
                actor_batch = valid_idx[actor_batch]
                critic_batch_vals = new_vals[critic_batch]
                actor_batch_adv = adv[actor_batch]  # adv = gae_val - val

                # update critic network
                critic_s = self.replay_buffer.get_all("states", critic_batch)
                curr_critic_loss = self._update_critic(critic_s, critic_batch_vals)

                # update actor network
                actor_s = self.replay_buffer.get("states", actor_batch[:,0])  # 必须得有goal,不然怎么mimic?
                actor_a = self.replay_buffer.get("actions", actor_batch[:,0])
                actor_logp = self.replay_buffer.get("logps", actor_batch[:,0])

                curr_actor_loss, curr_actor_clip_frac = self._update_actor(actor_s, actor_a, actor_logp, actor_batch_adv)
                # print("[train log] sub epoch %d loss = %.3f" % (b, curr_actor_loss))
                assert np.isfinite(curr_actor_loss).all() == True

                critic_loss += curr_critic_loss
                actor_loss += np.abs(curr_actor_loss)

        total_batches = mini_batches * self.epochs
        critic_loss /= total_batches
        actor_loss /= total_batches
        actor_clip_frac /= total_batches

        critic_loss = MPIUtil.reduce_avg(critic_loss)
        actor_loss = MPIUtil.reduce_avg(actor_loss)
        actor_clip_frac = MPIUtil.reduce_avg(actor_clip_frac)

        critic_stepsize = self.critic_solver.get_stepsize()
        actor_stepsize = self.update_actor_stepsize(actor_clip_frac)

        self.logger.log_tabular('Critic_Loss', critic_loss)
        self.logger.log_tabular('Critic_Stepsize', critic_stepsize)
        self.logger.log_tabular('Actor_Loss', actor_loss) 
        self.logger.log_tabular('Actor_Stepsize', actor_stepsize)
        self.logger.log_tabular('Clip_Frac', actor_clip_frac)
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
            self._update_mode()
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
        self.path.goals.append(g)
        self.path.terminate = self.world.env.check_terminate(self.id)

        return
    def _store_path(self, path):
        path_id = self.replay_buffer.store(path)
        valid_path = path_id != MathUtil.INVALID_IDX



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
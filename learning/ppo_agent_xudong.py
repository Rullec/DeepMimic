import numpy as np
import copy as copy
import tensorflow as tf

from util.logger import Logger
import util.mpi_util as MPIUtil
import util.math_util as MathUtil
import learning.rl_util as RLUtil
from env.env import Env
from learning.solvers.mpi_solver import MPISolver
import learning.nets.net_builder as NetBuilder
from learning.replay_buffer import ReplayBuffer
from learning.agent_xudong import Agent_xudong

'''
    PPOAgent_xudong, inherited from agent_xudong
'''
para_get = lambda key_name, default, json: default if (key_name not in json) else json[key_name]      

class PPOAgent_xudong(Agent_xudong):
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
    BATCHSIZE = "BatchSize"
    MINI_BATCHSIZE = "MiniBatchSize"
    REPLAYBUFFER_SIZE = "ReplayBufferSize"
    RATIO_CLIP = "RatioClip"
    TD_LAMBDA = "TDLambda"
    TEST_EPISODES = "TestEpisodes"

    def __init__(self, world, id, json_data):
        super.__init__(world, id)
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
        super.load_params(json_data)
        
        # read hyper paras from config
        self.discount = para_get(self.DISCOUNT, 0.95, json_data)
        self.batch_size = para_get(self.BATCHSIZE, 1024, json_data)
        self.minibatch_size = para_get(self.MINI_BATCHSIZE, 32, json_data)
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
        loss = tf.reduce_sum(tf.square(min_part, axis = -1) + tf.square(max_part, axis = -1))
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
        self.critic_grad_tf = tf.gradients(self.critic_loss_tf, critic_opt)
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
        num_procs = MPIUtil.get_num_procs()
        buffer_size = int(buffer_size / num_procs)
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size)
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
import numpy as np
import copy as copy
import tensorflow as tf

from learning.tf.pg_agent import PGAgent
from learning.tf.solvers.mpi_solver import MPISolver
import learning.tf.tf_util as TFUtil
import learning.tf.rl_util as RLUtil
from util.logger import Logger
import util.mpi_util as MPIUtil
import util.math_util as MathUtil
from env.env import Env

"""
Proximal Policy Optimization Agent
"""


class PPOAgent(PGAgent):
    NAME = "PPO"
    EPOCHS_KEY = "Epochs"
    BATCH_SIZE_KEY = "BatchSize"
    RATIO_CLIP_KEY = "RatioClip"
    NORM_ADV_CLIP_KEY = "NormAdvClip"
    TD_LAMBDA_KEY = "TDLambda"  # 肯定用了GAE
    TAR_CLIP_FRAC = "TarClipFrac"
    ACTOR_STEPSIZE_DECAY = "ActorStepsizeDecay"

    def __init__(self, world, id, json_data):
        super().__init__(world, id, json_data)
        return

    def _load_params(self, json_data):
        super()._load_params(json_data)

        self.epochs = (
            1 if (
                self.EPOCHS_KEY not in json_data) else json_data[self.EPOCHS_KEY]
        )
        self.batch_size = (
            1024
            if (self.BATCH_SIZE_KEY not in json_data)
            else json_data[self.BATCH_SIZE_KEY]
        )
        self.ratio_clip = (
            0.2
            if (self.RATIO_CLIP_KEY not in json_data)
            else json_data[self.RATIO_CLIP_KEY]
        )
        self.norm_adv_clip = (
            5
            if (self.NORM_ADV_CLIP_KEY not in json_data)
            else json_data[self.NORM_ADV_CLIP_KEY]
        )
        self.td_lambda = (
            0.95
            if (self.TD_LAMBDA_KEY not in json_data)
            else json_data[self.TD_LAMBDA_KEY]
        )
        self.tar_clip_frac = (
            -1
            if (self.TAR_CLIP_FRAC not in json_data)
            else json_data[self.TAR_CLIP_FRAC]
        )

        self.actor_stepsize_decay = (
            0.5
            if (self.ACTOR_STEPSIZE_DECAY not in json_data)
            else json_data[self.ACTOR_STEPSIZE_DECAY]
        )

        num_procs = MPIUtil.get_num_procs()
        local_batch_size = int(self.batch_size / num_procs)
        min_replay_size = 2 * local_batch_size  # needed to prevent buffer overflow
        assert self.replay_buffer_size > min_replay_size

        self.replay_buffer_size = np.maximum(
            min_replay_size, self.replay_buffer_size)

        return

    def _build_nets(self, json_data):
        """
            这里是ppo agent的build_net过程

            对于一个agent的action，必然是从一个net中来的，即:
                N(s) = a
            这个网络需要为当前的agent建立。当前的强化学习普遍使用actor-critic来加速训练。
                这并不只存在于PPO中;所谓actor-critic就是训练两个网络，其中critic的表现更加好一点，多迭代几次；
                这个critic训练的一般是value function，而actor训练的就是action。
                然后例如critic用TD训练，而actor用PPO/PG训练。
            这个函数就是建立了actor-critic网络，并且在他们的输出上加了高斯噪声等一系列的东西。

        """
        assert self.ACTOR_NET_KEY in json_data
        assert self.CRITIC_NET_KEY in json_data

        # actor 网络名字
        actor_net_name = json_data[self.ACTOR_NET_KEY]
        critic_net_name = json_data[self.CRITIC_NET_KEY]
        actor_init_output_scale = (
            1
            if (self.ACTOR_INIT_OUTPUT_SCALE_KEY not in json_data)
            else json_data[self.ACTOR_INIT_OUTPUT_SCALE_KEY]
        )

        # state多大? goal多大? action多大? 这些量后来都用来设置placeholder
        s_size = self.get_state_size()
        g_size = self.get_goal_size()
        a_size = self.get_action_size()

        # setup input tensors
        self.s_tf = tf.placeholder(
            tf.float32, shape=[None, s_size], name="s"
        )  # 输入state
        self.a_tf = tf.placeholder(
            tf.float32, shape=[None, a_size], name="a"
        )  # 输入action
        self.tar_val_tf = tf.placeholder(
            tf.float32, shape=[None], name="tar_val"
        )  # 输入:target value，从MC方法来的，外部计算输入，用于监督训练critic sa
        self.adv_tf = tf.placeholder(
            tf.float32, shape=[None], name="adv"
        )  # advantage function
        self.g_tf = tf.placeholder(
            tf.float32, shape=([None, g_size] if self.has_goal() else None), name="g"
        )  # goal
        self.old_logp_tf = tf.placeholder(
            tf.float32, shape=[None], name="old_logp"
        )  # old logp，用来做重要性采样
        self.exp_mask_tf = tf.placeholder(
            tf.float32, shape=[None], name="exp_mask"
        )  # 这个mask是干啥的?

        with tf.variable_scope("main"):
            with tf.variable_scope("actor"):
                # 建立actor网络
                # 输出是action所服从的高斯分布的均值。
                self.a_mean_tf = self._build_net_actor(
                    actor_net_name, actor_init_output_scale
                )
            with tf.variable_scope("critic"):
                # 建立critic网络
                self.critic_tf = self._build_net_critic(critic_net_name)

        if self.a_mean_tf != None:
            Logger.print("Built actor sa: " + actor_net_name)

        if self.critic_tf != None:
            Logger.print("Built critic sa: " + critic_net_name)

        # normalized action std = N(0, 0.05)
        self.norm_a_std_tf = self.exp_params_curr.noise * tf.ones(a_size)

        norm_a_noise_tf = self.norm_a_std_tf * tf.random_normal(
            shape=tf.shape(self.a_mean_tf)
        )

        # normalized action std = N(0, 0.05) * exp_mask
        norm_a_noise_tf *= tf.expand_dims(self.exp_mask_tf, axis=-1)
        self.norm_a_noise_tf = norm_a_noise_tf

        # action = a_mean + N(0, 0.05) * exp_mask * action_normalizer_std
        self.sample_a_tf = (
            self.a_mean_tf + norm_a_noise_tf * self.a_norm.std_tf
        )

        # P(action) = P(x = N(0, 0.05) * mask, mean = 0, std = N(0, 0.05))
        self.sample_a_logp_tf = TFUtil.calc_logp_gaussian(
            x_tf=norm_a_noise_tf, mean_tf=None, std_tf=self.norm_a_std_tf
        )

        return

    def _build_losses(self, json_data):
        actor_weight_decay = (
            0
            if (self.ACTOR_WEIGHT_DECAY_KEY not in json_data)
            else json_data[self.ACTOR_WEIGHT_DECAY_KEY]
        )
        critic_weight_decay = (
            0
            if (self.CRITIC_WEIGHT_DECAY_KEY not in json_data)
            else json_data[self.CRITIC_WEIGHT_DECAY_KEY]
        )

        # 这个val norm是干什么的?
        norm_val_diff = self.val_norm.normalize_tf(
            self.tar_val_tf
        ) - self.val_norm.normalize_tf(self.critic_tf)
        self.critic_loss_tf = 0.5 * tf.reduce_mean(
            tf.square(norm_val_diff)
        )  # critic loss1 = MSE

        if critic_weight_decay != 0:  # ciritc loss2 = critic网络参数尽可能接近0(绝对值小)
            self.critic_loss_tf += critic_weight_decay * self._weight_decay_loss(
                "main/critic"
            )

        # action的范围究竟是不是角度限制?
        norm_tar_a_tf = self.a_norm.normalize_tf(
            self.a_tf)  # 外部输入的action变化到分布上
        self._norm_a_mean_tf = self.a_norm.normalize_tf(
            self.a_mean_tf
        )  # 把网络声称，也变化到这个分布上。

        # 计算重要性采样分子的:当前policy下的占位符输入action 的概率。
        self.logp_tf = TFUtil.calc_logp_gaussian(
            norm_tar_a_tf, self._norm_a_mean_tf, self.norm_a_std_tf
        )
        ratio_tf = tf.exp(self.logp_tf - self.old_logp_tf)  # ratio的实现
        self.ratio_tf = ratio_tf
        actor_loss0 = self.adv_tf * ratio_tf  # 第一个loss: 优势函数 * ratio
        actor_loss1 = self.adv_tf * tf.clip_by_value(
            ratio_tf, 1.0 - self.ratio_clip, 1 + self.ratio_clip
        )  # 第二个loss: 裁剪 * advantage
        # 所以说，我强烈怀疑是输入的advantage有问题，这就去排查一下。

        # 这里定义了loss
        self.actor_loss_tf = - \
            tf.reduce_mean(tf.minimum(actor_loss0, actor_loss1))

        # action有上下界
        # print("action bound min = " % self.a_bound_min)
        # print("action bound max = " % self.a_bound_max)
        norm_a_bound_min = self.a_norm.normalize(self.a_bound_min)
        norm_a_bound_max = self.a_norm.normalize(self.a_bound_max)
        a_bound_loss = TFUtil.calc_bound_loss(
            self._norm_a_mean_tf, norm_a_bound_min, norm_a_bound_max
        )
        self.actor_loss_tf += a_bound_loss

        if actor_weight_decay != 0:
            self.actor_loss_tf += actor_weight_decay * self._weight_decay_loss(
                "main/actor"
            )

        # for debugging
        self.clip_frac_tf = tf.reduce_mean(
            tf.to_float(tf.greater(tf.abs(ratio_tf - 1.0), self.ratio_clip))
        )

        return

    def _build_solvers(self, json_data):
        actor_stepsize = (
            0.001
            if (self.ACTOR_STEPSIZE_KEY not in json_data)
            else json_data[self.ACTOR_STEPSIZE_KEY]
        )
        actor_momentum = (
            0.9
            if (self.ACTOR_MOMENTUM_KEY not in json_data)
            else json_data[self.ACTOR_MOMENTUM_KEY]
        )
        critic_stepsize = (
            0.01
            if (self.CRITIC_STEPSIZE_KEY not in json_data)
            else json_data[self.CRITIC_STEPSIZE_KEY]
        )
        critic_momentum = (
            0.9
            if (self.CRITIC_MOMENTUM_KEY not in json_data)
            else json_data[self.CRITIC_MOMENTUM_KEY]
        )

        critic_vars = self._tf_vars("main/critic")
        critic_opt = tf.train.MomentumOptimizer(
            learning_rate=critic_stepsize, momentum=critic_momentum
        )
        self.critic_grad_tf = tf.gradients(self.critic_loss_tf, critic_vars)
        self.critic_solver = MPISolver(self.sess, critic_opt, critic_vars)

        self._actor_stepsize_tf = tf.get_variable(
            dtype=tf.float32,
            name="actor_stepsize",
            initializer=actor_stepsize,
            trainable=False,
        )
        self._actor_stepsize_ph = tf.get_variable(
            dtype=tf.float32, name="actor_stepsize_ph", shape=[]
        )
        self._actor_stepsize_update_op = self._actor_stepsize_tf.assign(
            self._actor_stepsize_ph
        )

        actor_vars = self._tf_vars("main/actor")
        actor_opt = tf.train.MomentumOptimizer(
            learning_rate=self._actor_stepsize_tf, momentum=actor_momentum
        )
        self.actor_grad_tf = tf.gradients(self.actor_loss_tf, actor_vars)
        self.actor_solver = MPISolver(self.sess, actor_opt, actor_vars)

        return

    def _decide_action(self, s, g):
        assert np.isfinite(s).all() == True

        with self.sess.as_default(), self.graph.as_default():
            self._exp_action = self._enable_stoch_policy() and MathUtil.flip_coin(
                self.exp_params_curr.rate
            )
            # print(f"[decide action] exp_action_mask = {self._exp_action}")
            a, logp, a_mean = self._eval_actor(s, g, self._exp_action)
        return a[0], logp[0], a_mean[0]

    def _eval_actor(self, s, g, enable_exp):
        s = np.reshape(s, [-1, self.get_state_size()])
        g = np.reshape(g, [-1, self.get_goal_size()]
                       ) if self.has_goal() else None

        feed = {
            self.s_tf: s,
            self.g_tf: g,
            self.exp_mask_tf: np.array([1 if enable_exp else 0]),
        }

        a, logp, a_mean = self.sess.run(
            [self.sample_a_tf, self.sample_a_logp_tf,
                self.a_mean_tf], feed_dict=feed
        )

        # print(f"[debug] ppo action {a}")
        return a, logp, a_mean

    def _train_step(self):
        """
            for each substep in update_agents, after decide and apply the selected action

            there will be several again train steps (several iters)
                to update network weight

        :return:
        """
        adv_eps = 1e-5

        # get the buffer info, st to ed
        start_idx = self.replay_buffer.buffer_tail
        end_idx = self.replay_buffer.buffer_head
        assert start_idx == 0
        assert (
            self.replay_buffer.get_current_size() <= self.replay_buffer.buffer_size
        )  # must avoid overflow
        assert start_idx < end_idx

        # buffer idx array
        idx = np.array(list(range(start_idx, end_idx)))
        end_mask = self.replay_buffer.is_path_end(idx)
        end_mask = np.logical_not(end_mask)

        vals = self._compute_batch_vals(start_idx, end_idx)
        new_vals = self._compute_batch_new_vals(start_idx, end_idx, vals)

        valid_idx = idx[end_mask]
        exp_idx = self.replay_buffer.get_idx_filtered(
            self.EXP_ACTION_FLAG).copy()
        num_valid_idx = valid_idx.shape[0]
        num_exp_idx = exp_idx.shape[0]
        exp_idx = np.column_stack(
            [exp_idx, np.array(list(range(0, num_exp_idx)), dtype=np.int32)]
        )

        local_sample_count = valid_idx.size
        global_sample_count = int(MPIUtil.reduce_sum(local_sample_count))
        mini_batches = int(np.ceil(global_sample_count / self.mini_batch_size))

        adv = new_vals[exp_idx[:, 0]] - vals[exp_idx[:, 0]]
        new_vals = np.clip(new_vals, self.val_min, self.val_max)

        adv_mean = np.mean(adv)
        adv_std = np.std(adv)
        adv = (adv - adv_mean) / (adv_std + adv_eps)
        adv = np.clip(adv, -self.norm_adv_clip, self.norm_adv_clip)

        if np.isfinite(adv).all() == False:
            print(f"[error] advantage include Nan or Inf")
            raw_adv = new_vals[exp_idx[:, 0]] - vals[exp_idx[:, 0]]

            print(f"start idx {start_idx} end idx {end_idx}")
            print("new val = ", new_vals[exp_idx[:, 0]])
            print("old val = ", vals[exp_idx[:, 0]])
            print("raw adv = ", raw_adv)
            print("now adv = ", adv)
            print("adv mean = ", adv_mean)
            print("adv std = ", adv_std)
            print("norm adv clip ", self.norm_adv_clip)

        critic_loss = 0
        actor_loss = 0
        actor_clip_frac = 0

        for e in range(self.epochs):
            # 对于每个epoch，先把idx shuffle
            np.random.shuffle(valid_idx)
            np.random.shuffle(exp_idx)

            for b in range(mini_batches):
                # 又有minibatch
                batch_idx_beg = (
                    b * self._local_mini_batch_size
                )  # 逐个步进的取batch(shuffle后顺序拿)
                batch_idx_end = batch_idx_beg + self._local_mini_batch_size

                critic_batch = np.array(
                    range(batch_idx_beg, batch_idx_end), dtype=np.int32
                )
                actor_batch = critic_batch.copy()
                critic_batch = np.mod(critic_batch, num_valid_idx)
                actor_batch = np.mod(actor_batch, num_exp_idx)
                shuffle_actor = (actor_batch[-1] < actor_batch[0]) or (
                    actor_batch[-1] == num_exp_idx - 1
                )

                critic_batch = valid_idx[critic_batch]
                actor_batch = exp_idx[actor_batch]
                critic_batch_vals = new_vals[critic_batch]
                actor_batch_adv = adv[actor_batch[:, 1]]

                critic_s = self.replay_buffer.get("states", critic_batch)
                critic_g = (
                    self.replay_buffer.get("goals", critic_batch)
                    if self.has_goal()
                    else None
                )

                # update critic
                curr_critic_loss = self._update_critic(
                    critic_s, critic_g, critic_batch_vals
                )

                actor_s = self.replay_buffer.get("states", actor_batch[:, 0])
                actor_g = (
                    self.replay_buffer.get("goals", actor_batch[:, 0])
                    if self.has_goal()
                    else None
                )  # 必须得有goal,不然怎么mimic?
                actor_a = self.replay_buffer.get("actions", actor_batch[:, 0])
                actor_logp = self.replay_buffer.get("logps", actor_batch[:, 0])

                # update actor
                # assert actor_g is not None
                # print("advantage = %s" % str(adv))
                # assert np.isfinite(adv).all() == True
                # raise(ValueError)
                curr_actor_loss, curr_actor_clip_frac = self._update_actor(
                    actor_s, actor_g, actor_a, actor_logp, actor_batch_adv
                )
                # print("[train log] sub epoch %d loss = %.3f" % (b, curr_actor_loss))
                assert np.isfinite(curr_actor_loss).all() == True

                # actor loss是Nan
                critic_loss += curr_critic_loss
                actor_loss += np.abs(curr_actor_loss)
                actor_clip_frac += curr_actor_clip_frac

                if shuffle_actor:
                    np.random.shuffle(exp_idx)

        total_batches = mini_batches * self.epochs
        critic_loss /= total_batches
        actor_loss /= total_batches
        actor_clip_frac /= total_batches

        critic_loss = MPIUtil.reduce_avg(critic_loss)
        actor_loss = MPIUtil.reduce_avg(actor_loss)
        actor_clip_frac = MPIUtil.reduce_avg(actor_clip_frac)

        critic_stepsize = self.critic_solver.get_stepsize()
        actor_stepsize = self.update_actor_stepsize(actor_clip_frac)

        self.logger.log_tabular("Critic_Loss", critic_loss)
        self.logger.log_tabular("Critic_Stepsize", critic_stepsize)
        self.logger.log_tabular("Actor_Loss", actor_loss)
        self.logger.log_tabular("Actor_Stepsize", actor_stepsize)
        self.logger.log_tabular("Clip_Frac", actor_clip_frac)
        self.logger.log_tabular("Adv_Mean", adv_mean)
        self.logger.log_tabular("Adv_Std", adv_std)

        # if we want to save buffer in TRAIN mode
        # then we will save buffer to disk at here,
        # just before clearing the buffer
        if self.buffer_save_type == self.BufferSaveType.TRAIN:
            self.replay_buffer.save()

        self.replay_buffer.clear()

        return

    def _get_iters_per_update(self):
        return 1

    def _valid_train_step(self):
        # 只有当批量达到batch size的时候才行
        samples = self.replay_buffer.get_current_size()
        exp_samples = self.replay_buffer.count_filtered(self.EXP_ACTION_FLAG)
        global_sample_count = int(MPIUtil.reduce_sum(samples))
        global_exp_min = int(MPIUtil.reduce_min(exp_samples))
        # print("[valid train step] current %d samples > batchsize = %d" % (global_sample_count, self.batch_size))
        return (global_sample_count > self.batch_size) and (global_exp_min > 0)

    def _compute_batch_vals(self, start_idx, end_idx):
        states = self.replay_buffer.get_all("states")[start_idx:end_idx]
        goals = (
            self.replay_buffer.get_all("goals")[start_idx:end_idx]
            if self.has_goal()
            else None
        )

        idx = np.array(list(range(start_idx, end_idx)))
        is_end = self.replay_buffer.is_path_end(idx)
        is_fail = self.replay_buffer.check_terminal_flag(
            idx, Env.Terminate.Fail)
        is_succ = self.replay_buffer.check_terminal_flag(
            idx, Env.Terminate.Succ)
        is_fail = np.logical_and(is_end, is_fail)
        is_succ = np.logical_and(is_end, is_succ)

        vals = self._eval_critic(states, goals)
        vals[is_fail] = self.val_fail
        vals[is_succ] = self.val_succ
        if np.isfinite(vals).all() == False:
            print("[error] compute batch vals inf or nan!")
            print(f"val fail = {self.val_fail} val succ {self.val_succ}")
            print(f"states = {states}")
            print(f"vals = {vals}")
            exit(0)
        return vals

    def _compute_batch_new_vals(self, start_idx, end_idx, val_buffer):
        rewards = self.replay_buffer.get_all("rewards")[start_idx:end_idx]

        if self.discount == 0:
            new_vals = rewards.copy()
        else:
            new_vals = np.zeros_like(val_buffer)

            curr_idx = start_idx
            while curr_idx < end_idx:
                idx0 = curr_idx - start_idx
                idx1 = self.replay_buffer.get_path_end(curr_idx) - start_idx
                r = rewards[idx0:idx1]
                v = val_buffer[idx0: (idx1 + 1)]

                new_vals[idx0:idx1] = RLUtil.compute_return(
                    r, self.discount, self.td_lambda, v
                )
                curr_idx = idx1 + start_idx + 1

        return new_vals

    def _update_critic(self, s, g, tar_vals):
        feed = {self.s_tf: s, self.g_tf: g, self.tar_val_tf: tar_vals}

        loss, grads = self.sess.run(
            [self.critic_loss_tf, self.critic_grad_tf], feed)

        def compute_max_abs(grads):
            assert type(grads) == list
            cur_max = 0
            for i in grads:
                assert(type(i) == np.ndarray)
                cur_max = max(np.max(np.abs(i)), cur_max)
            return cur_max

        # grad clip
        clip_threshold = 100
        old_max = compute_max_abs(grads)
        if old_max > clip_threshold:
            for i in range(len(grads)):
                grads[i] = np.clip(grads[i], -clip_threshold, clip_threshold)
            print(
                f"[warn] cliped max critic grad abs from {old_max} to {clip_threshold}")

        self.critic_solver.update(grads)  # 这个是更新critic...
        return loss

    def _update_actor(self, s, g, a, logp, adv):
        # 保证更新actor的所有值就不能有Nan
        # 如果他们有问题，那一定是仿真环境出了问题
        assert np.isfinite(s).all() == True

        assert np.isfinite(a).all() == True
        assert np.isfinite(logp).all() == True
        assert np.isfinite(adv).all() == True
        # assert np.isfinite(g).all() == True # 这个目标有问题

        feed = {
            self.s_tf: s,
            self.g_tf: g,
            self.a_tf: a,
            self.adv_tf: adv,
            self.old_logp_tf: logp,
        }
        """        norm_x = (x - self.mean_tf) / self.std_tf
        norm_x = tf.clip_by_value(norm_x, -self.clip, self.clip)
        """
        # 获取loss, 梯度, clip_frac?
        # a_mean, norm_a_mean, a_norm_mean, a_norm_std, a_norm_clip = self.sess.run([self.a_mean_tf, self._norm_a_mean_tf,
        #     self.a_norm.mean_tf, self.a_norm.std_tf, self.a_norm.clip], feed_dict=feed)
        a_mean, norm_a_mean = self.sess.run(
            [self.a_mean_tf, self._norm_a_mean_tf, ], feed_dict=feed
        )
        # print("a_mean = %s" % str(a_mean))
        # print("a_norm_mean = %s" % str(self.sess.run(self.a_norm.mean_tf)))
        # print("a_norm_std = %s" % str(self.sess.run(self.a_norm.std_tf)))
        # print("a_norm_clip = %s" % str(self.a_norm.clip))

        # print("old p = %s" % str(logp))
        # print("norm a mean = %s" % str(norm_a_mean))

        assert np.isfinite(norm_a_mean).all() == True

        """
        self.norm_a_std_tf = self.exp_params_curr.noise * tf.ones(a_size)

        norm_a_noise_tf = self.norm_a_std_tf * tf.random_normal(shape=tf.shape(self.a_mean_tf))
        norm_a_noise_tf *= tf.expand_dims(self.exp_mask_tf, axis=-1)
        self.norm_a_noise_tf = norm_a_noise_tf
        self.sample_a_tf = self.a_mean_tf + norm_a_noise_tf * self.a_norm.std_tf    # action采样输出, 是mean ->无穷的值 + 噪音
        self.sample_a_logp_tf = TFUtil.calc_logp_gaussian(x_tf=norm_a_noise_tf, mean_tf=None, std_tf=self.norm_a_std_tf)    # 他对应的当前概率是这个

        """
        loss, grads, clip_frac, ratio_all, log_p_new = self.sess.run(
            [
                self.actor_loss_tf,
                self.actor_grad_tf,
                self.clip_frac_tf,
                self.ratio_tf,
                self.logp_tf,
            ],
            feed,
        )

        # print("ratio(old_prob/new_prob) = %s " % str(ratio_all))
        # print("ratio clip = %s " % str(self.ratio_clip))
        # print("clip frac(exceed prohibited) = %s " % str(clip_frac))
        # print("new logp = %s " % str(log_p_new))
        assert np.isfinite(ratio_all).all() == True
        assert np.isfinite(log_p_new).all() == True

        self.actor_solver.update(grads)

        return loss, clip_frac

    # clip_frac \in [0-1] , clip_frac
    def update_actor_stepsize(self, clip_frac):
        clip_tol = 1.5
        step_scale = 2
        max_stepsize = 1e-2
        min_stepsize = 1e-8
        warmup_iters = 5

        actor_stepsize = self.actor_solver.get_stepsize()
        # target clip fraction: if the clip fraction is too big, the training will be harder
        if self.tar_clip_frac >= 0 and self.iter > warmup_iters:
            min_clip = self.tar_clip_frac / clip_tol
            max_clip = self.tar_clip_frac * clip_tol
            under_tol = clip_frac < min_clip
            over_tol = clip_frac > max_clip

            if over_tol or under_tol:
                if over_tol:
                    actor_stepsize *= self.actor_stepsize_decay
                else:
                    actor_stepsize /= self.actor_stepsize_decay

                actor_stepsize = np.clip(
                    actor_stepsize, min_stepsize, max_stepsize)
                self.set_actor_stepsize(actor_stepsize)

        return actor_stepsize

    def set_actor_stepsize(self, stepsize):
        feed = {
            self._actor_stepsize_ph: stepsize,
        }
        self.sess.run(self._actor_stepsize_update_op, feed)
        return

    def _update_new_action(self):
        pass
        """
            when the agent need a new action, this function will be called.

        :return:
        """
        # 获取新的action
        s = self._record_state()
        # c = self._record_contact_info()
        # p = self._record_pose()
        g = self._record_goal()
        # print("goal is %s" % str(g))
        # exit()

        if not (self._is_first_step()):
            r = self._record_reward()
            # print("reward : " + str(r))
            self.path.rewards.append(r)

            if self._enable_draw():
                self.log_reward(r)
            try:
                assert np.isfinite(r).all() == True
            except:
                print("some reward is Nan!, r = %s" % str(r))

        try:
            assert np.isfinite(s).all() == True
        except:
            print("some state is Nan!, s = %s" % str(s))

        a, logp, a_mean = self._decide_action(s=s, g=g)
        # diff = (a - a_mean)
        # print(f"action diff = {np.linalg.norm(diff)}")
        # print(
        #     f"[check] state norm {np.linalg.norm(s)} action_mean norm {np.linalg.norm(a_mean)} action norm {np.linalg.norm(a)}")
        assert len(np.shape(a)) == 1
        assert len(np.shape(logp)) <= 1

        flags = self._record_flags()
        # 应用action
        # reward并不马上给出，而是在下次apply action的时候得到
        try:
            assert np.isfinite(a).all() == True
        except:
            print("some action is Nan!, a = %s" % str(a))
        self._apply_action(a)
        self.path.states.append(s)
        # self.path.contact_info.append(c)
        # self.path.poses.append(p)
        self.path.goals.append(g)
        self.path.actions.append(a)
        self.path.logps.append(logp)
        self.path.flags.append(flags)
        self.path.action_mean.append(a_mean)

        # if self._enable_draw():
        #     self._log_val(s, g)

        return

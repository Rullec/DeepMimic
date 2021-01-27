import numpy as np
import copy
import os
import time
import json
import datetime
from env.action_space import ActionSpace
from abc import ABC, abstractmethod
from enum import Enum
from learning.torch.path_torch import PathTorch
from learning.torch.nets.net_builder import build_net
import torch
import util.mpi_util as MPIUtil
import torch.optim as optim
from learning.torch.replay_buffer_torch import ReplayBufferTorch
from util.logger import Logger
from learning.tf.exp_params import ExpParams
from learning.torch.torch_normalizer import NormalizerTorch


class TorchAgent:
    """
        Torch Agent
    """
    NAME = "DiffMBRL"
    POLICY_NET_KEY = "PolicyNet"
    LEARNINGRATE_KEY = "LearningRate"
    LEARNINGRATE_DECAY_KEY = "LearningRateDecay"
    REPLAY_BUFFER_CAPACITY_KEY = "ReplayBufferCapacity"
    MAX_SAMPLES_KEY = "MaxSamples"
    EXP_ANNEAL_SAMPLES_KEY = "ExpAnnealSamples"
    EXP_PARAM_BEG_KEY = "ExpParamsBeg"
    EXP_PARAM_END_KEY = "ExpParamsEnd"
    WEIGHT_LOSS_KEY = "WeightLoss"
    ENABLE_UPDATE_NORMALIZERS_KEY = "EnableUpdateNormalizers"
    INIT_NORMALIZER_SAMPLES = "InitNormalizerSamples"
    NORMALIZER_ALPHA_KEY = "NormalizerAlpha"
    ACTIVATION_KEY = "Activation"
    ACTION_NORMALIZER_KEY = "ActionNormalizer"
    STATE_NORMALIZER_KEY = "StateNormalizer"
    DROPOUT_KEY = "Dropout"
    ENABLE_ACTION_KEY = "EnableActionNoise"
    TEST_GAP_KEY = "TestGap"
    TEST_EPISODE_KEY = "TestEpisode"

    class Mode(Enum):
        TRAIN = 0
        TEST = 1
        TRAIN_END = 2

    def __init__(self, world, id, json_data):
        """
            Init method, create from json data
        """
        # 1. hyperparameters init value
        self.replay_buffer_capacity = 400
        self.lr = 1e-3
        self.lr_decay = 1.0
        self.exp_anneal_samples = 5e5
        self.max_samples = 5e5
        self.weight_loss = 0.
        self.enable_action_noise = False
        self.test_episodes = int(0)
        self.test_gap = int(0)
        self.exp_params_beg = ExpParams()
        self.exp_params_end = ExpParams()
        self.exp_params_curr = ExpParams()
        self.enable_update_normalizer = True
        # only when total samples is lower than "init_normalizer_samples" will the normalizers being updated
        self.init_normalizer_samples = 0
        self._train_iters = 0
        # 2. runtime vars init value
        self.world = world
        self.logger = Logger()
        self.test_return = 0
        self.test_episode_count = int(0)
        self._enable_training = True
        self.id = id
        self._mode = self.Mode.TRAIN
        self._begin_time = time.time()
        self._total_sample_count = 0
        self.output_dir = "output/0111/test"
        self.path = PathTorch()
        self.replay_buffer = ReplayBufferTorch(self.replay_buffer_capacity)

        # 3. hyperparams from agent
        self._load_params(json_data)

        # 4. build agent graph
        self._build_graph(json_data)

        # 5. build loss
        self._build_loss()

        return

    def _load_params(self, json_data):
        '''
            Load the hyperparameters from agent config
        '''
        assert self.LEARNINGRATE_KEY in json_data
        self.lr = json_data[self.LEARNINGRATE_KEY]

        assert self.LEARNINGRATE_DECAY_KEY in json_data
        self.lr_decay = json_data[self.LEARNINGRATE_DECAY_KEY]

        assert self.REPLAY_BUFFER_CAPACITY_KEY in json_data
        self.replay_buffer_capacity = json_data[self.REPLAY_BUFFER_CAPACITY_KEY]
        self.replay_buffer.capacity = self.replay_buffer_capacity

        assert self.MAX_SAMPLES_KEY in json_data
        self.max_samples = json_data[self.MAX_SAMPLES_KEY]

        assert self.EXP_ANNEAL_SAMPLES_KEY in json_data
        self.exp_anneal_samples = json_data[self.EXP_ANNEAL_SAMPLES_KEY]

        assert self.EXP_PARAM_BEG_KEY in json_data
        self.exp_params_beg.load(json_data[self.EXP_PARAM_BEG_KEY])

        assert self.EXP_PARAM_END_KEY in json_data
        self.exp_params_end.load(json_data[self.EXP_PARAM_END_KEY])

        assert self.WEIGHT_LOSS_KEY in json_data
        self.weight_loss = json_data[self.WEIGHT_LOSS_KEY]

        assert self.ENABLE_UPDATE_NORMALIZERS_KEY in json_data
        self.enable_update_normalizer = json_data[self.ENABLE_UPDATE_NORMALIZERS_KEY]

        assert self.INIT_NORMALIZER_SAMPLES in json_data
        self.init_normalizer_samples = json_data[self.INIT_NORMALIZER_SAMPLES]

        assert self.NORMALIZER_ALPHA_KEY in json_data
        self.normalizer_alpha = json_data[self.NORMALIZER_ALPHA_KEY]

        assert self.ENABLE_ACTION_KEY in json_data
        self.enable_action_noise = json_data[self.ENABLE_ACTION_KEY]

        assert self.TEST_GAP_KEY in json_data
        self.test_gap = json_data[self.TEST_GAP_KEY]

        assert self.TEST_EPISODE_KEY in json_data
        self.test_episodes = json_data[self.TEST_EPISODE_KEY]

        self.exp_params_curr = copy.deepcopy(self.exp_params_beg)
        return

    def _get_parameters(self):
        return self.action.parameters()

    def _update_optimizer(self):
        self.optimizer = optim.Adam(
            self._get_parameters(), lr=self.lr, weight_decay=self.weight_loss)
        print("-------update optimizer to adam-------")

    def _build_loss(self):
        # self._get_parameters()
        self.optimizer = optim.SGD(
            self._get_parameters(), lr=self.lr, weight_decay=self.weight_loss)

    def save_model(self, out_path):
        tar_dir = os.path.dirname(out_path)
        if False == os.path.exists(tar_dir):
            os.makedirs(tar_dir)
        save_state_dict = self.action.state_dict()
        save_state_dict[self.STATE_NORMALIZER_KEY] = self.state_normalizer
        save_state_dict[self.ACTION_NORMALIZER_KEY] = self.action_normalizer
        torch.save(save_state_dict, out_path)

        print(f"[log] torch save model to {out_path}")

        return

    def load_old_model(self, state_dict):
        new_old_map = [
            ("layers.0", "input"),
            ("layers.1", "fc1"),
            ("layers.2", "fc2"),
        ]

        type_name = ["weight", "bias"]
        # print("old")
        for new_key, old_key in new_old_map:
            for type_key in type_name:
                new_total_key = f"{new_key}.{type_key}"
                old_total_key = f"{old_key}.{type_key}"
                assert old_total_key in state_dict
                value = state_dict[old_total_key]
                state_dict.pop(old_total_key)
                state_dict[new_total_key] = value

        # assert False
        return state_dict

    def load_model(self, in_path):
        if os.path.exists(in_path) == False:
            return
        load_state_dict = torch.load(in_path)

        # old version model
        if "layers.0.weight" not in load_state_dict:
            load_state_dict = self.load_old_model(load_state_dict)

        if (self.STATE_NORMALIZER_KEY in load_state_dict) == False or (self.ACTION_NORMALIZER_KEY in load_state_dict) == False:
            print(f"[warn] no noramlizers found in given model {in_path}")
        else:
            self.state_normalizer = load_state_dict[self.STATE_NORMALIZER_KEY]
            self.action_normalizer = load_state_dict[self.ACTION_NORMALIZER_KEY]
            self.enable_update_normalizer = self.state_normalizer.sample_count < self.init_normalizer_samples
            load_state_dict.pop(self.STATE_NORMALIZER_KEY)
            load_state_dict.pop(self.ACTION_NORMALIZER_KEY)
            print(
                f"[debug] load state & action normalizer succ from {in_path}")
        self.action.load_state_dict(load_state_dict)
        print(f"[log] torch load model from {in_path}")
        print(f"a mean {self.action_normalizer.mean}")
        print(f"a std {self.action_normalizer.std}")
        print(f"s mean {self.state_normalizer.mean}")
        print(f"s std {self.state_normalizer.std}")
        # self.draw_param_hist()
        return

    def _enable_stoch_policy(self):
        # in training mode and
        # in test mode, no noise
        # in training mode but disable training, no noise
        return self.enable_training and (self._mode == self.Mode.TRAIN or self._mode == self.Mode.TRAIN_END) and self.enable_action_noise

    def _decide_action(self, s, g):
        a = self._infer_action(torch.Tensor(s)).detach().numpy()

        if self._enable_stoch_policy() is True:
            print(f"[debug] decide action, enable stoch polciy, raw a = {a}")
            import util.math_util as MathUtil
            whether_add_noise = MathUtil.flip_coin(self.exp_params_curr.rate)
            amptitude_noise = self.exp_params_curr.noise
            noise = amptitude_noise * whether_add_noise * \
                torch.randn(a.shape) * self.action_normalizer.std
            a = a + noise.detach().numpy()
            print(f"[debug] add action noise {noise} final action {a}")
        else:
            if self.enable_training == False:
                print("[debug] not training: disable stoch policy")
            else:
                print("[debug] enable training: disable stoch policy, no noise")
        return a

    def _infer_action(self, s):
        assert type(s) is torch.Tensor
        # if train is False:
        #     self.action.eval()
        # else:
        #     self.action.train()

        return self.action_normalizer.unnormalize(
            self.action(self.state_normalizer.normalize(s)))

    def _get_output_path(self):
        assert False
        return

    def _get_intermediate_output_path(self):
        assert False
        return

    def _train_step(self):
        assert False
        return

    def _get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']

    def _set_lr(self, lr):
        for g in self.optimizer.param_groups:
            if 'lr' in g:
                g['lr'] = lr

    def _update_exp_params(self):
        lerp = float(self._total_sample_count) / self.exp_anneal_samples

        lerp = np.clip(lerp, 0.0, 1.0)
        self.exp_params_curr = self.exp_params_beg.lerp(
            self.exp_params_end, lerp)

        print(
            f"[debug] action noise amptitude {self.exp_params_curr.noise} rate {self.exp_params_curr.rate}")
        return

    def _get_self_grad(self):
        # 1. prepare
        states = np.array(self.replay_buffer.get_state())
        drdas = np.array(self.replay_buffer.get_drda())

        actions_torch = self._infer_action(torch.Tensor(states))
        drdas_torch = torch.Tensor(drdas)

        # 2. get result
        action_size = self.get_action_size()
        num_param_group = len(
            list(self._get_parameters()))
        param_sizes = [i.shape for i in self._get_parameters()]

        drdtheta_total = []
        drdtheta_lst = [[] for i in range(num_param_group)]
        for n_step_idx in range(states.shape[0]):
            drda_step = torch.unsqueeze(drdas_torch[n_step_idx], 0)
            # print(f"drda shape {drda_step.shape}")
            dadtheta_lst = [[] for i in range(num_param_group)]
            for a_idx in range(action_size):
                # the gradient of this scale w.r.t to the group of parameters
                tmp_grad = torch.autograd.grad(
                    actions_torch[n_step_idx, a_idx], self._get_parameters(), retain_graph=True)
                assert len(tmp_grad) == num_param_group
                for _id, i in enumerate(tmp_grad):
                    assert i.shape == param_sizes[_id]
                    dadtheta_lst[_id].append(tmp_grad[_id])

                # assert tmp_grad.shape[0] == param_size, f"{tmp_grad.shape} != {param_size}"

            for _id, ele in enumerate(dadtheta_lst):
                dadtheta_id = torch.stack(dadtheta_lst[_id])

                if len(dadtheta_id.shape) == 3:
                    drdtheta = torch.einsum(
                        'ab,bcd->acd', drda_step, dadtheta_id)
                else:
                    drdtheta = torch.einsum(
                        'ab,bc->ac', drda_step, dadtheta_id)
                drdtheta_lst[_id].append(drdtheta)

        drdtheta_lst = [torch.reshape(-torch.mean(torch.squeeze(
            torch.stack(i)), axis=0), param_sizes[_idx]).detach() for _idx, i in enumerate(drdtheta_lst)]

        return drdtheta_lst

    def _grad_clip(self, lim=3):
        torch.nn.utils.clip_grad_value_(self._get_parameters(), lim)

        res = [i.grad for i in self._get_parameters()]
        max_res = max([np.max(np.array(i.detach())) for i in res])
        min_res = min([np.min(np.array(i.detach())) for i in res])
        print(f"max grad {max_res} min grad {min_res}")

    def _apply_self_grad(self):
        self_grads = self._get_self_grad()
        for _idx, param in enumerate(self._get_parameters()):
            param.grad = self_grads[_idx]

    def _train(self):
        """
            this function is called when the update_counters >= update_period
            train the current agent
        """
        # 1. construct the loss and backward
        self.optimizer.zero_grad()

        # 2. construct the loss and the gradient
        # self._apply_self_grad()
        states = np.array(self.replay_buffer.get_state())
        drdas = np.array(self.replay_buffer.get_drda())

        actions_torch = self._infer_action(
            torch.Tensor(states))
        drdas_torch = torch.Tensor(drdas)
        loss_sum = -torch.mean(drdas_torch *
                               actions_torch) * self.get_action_size()
        loss_sum.backward()

        self._grad_clip()

        # 3. do forward update, update the lr, sample_counts,
        self.optimizer.step()

        self._set_lr(max(self.lr_decay * self._get_lr(), 1e-6))
        self._total_sample_count += self.replay_buffer.get_cur_size()
        self.world.env.set_sample_count(self._total_sample_count)
        self._update_exp_params()

        # 4. output and clear
        output_name = datetime.datetime.now().strftime("%m-%d-%H:%M:%S")
        output_name = f"{output_name}-{str(self.replay_buffer.get_avg_reward())[:6]}.pkl"
        output_path = os.path.join(self.output_dir, output_name)
        self.save_model(output_path)

        cost_time = time.time() - self._begin_time
        avg_rew = self.replay_buffer.get_avg_reward()
        print(
            f"[log] total samples {self._total_sample_count} train time {cost_time} s, avg reward {avg_rew}, lr {self._get_lr()}")
        if self._total_sample_count > self.max_samples:
            print(f"[log] total samples exceed max {self.max_samples}, exit")
            exit(0)

        self._print_statistics()

        if self.enable_update_normalizer:
            self._update_normalizers()

        else:
            print("[update] normalizers kept!")
            print(f"Normalizer state mean = {self.state_normalizer.mean}")
            print(f"Normalizer state std = {self.state_normalizer.std}")
            print(f"Normalizer action mean = {self.action_normalizer.mean}")
            print(f"Normalizer action std = {self.action_normalizer.std}")

        # if self._train_iters == 200:
        #     self._update_optimizer()

        self.replay_buffer.clear()

        self._mode = self.Mode.TRAIN
        self._train_iters += 1

        # if current train iters is the same as
        print(
            f"[debug] train iters {self._train_iters} test gap {self.test_gap}")
        if (self._train_iters % self.test_gap) == 0:
            print(f"[debug] change mode from train to train end")
            self._mode = self.Mode.TRAIN_END

    def _update_normalizers(self):
        # update the normalizer
        # 1. get current states and actions
        states = np.array(self.replay_buffer.get_state())
        actions = np.array(self.replay_buffer.get_action())
        samples = self.replay_buffer.get_cur_size()

        # 2. caclulate the mean and std for them?
        state_mean = np.mean(states, axis=0)
        state_std = np.std(states, axis=0)
        self.state_normalizer.update(state_mean, state_std, samples)

        action_mean = np.mean(actions, axis=0)
        action_std = np.std(actions, axis=0)
        self.action_normalizer.update(action_mean, action_std, samples)

    def _print_statistics(self):
        # output the data dist
        np.set_printoptions(suppress=True)
        states = np.array(self.replay_buffer.get_state())
        s_mean = np.mean(states, axis=0)
        s_std = np.std(states, axis=0)
        print(f"[stat] state mean {s_mean}")
        print(f"[stat] state std {s_std}")
        actions = np.array(self.replay_buffer.get_action())
        a_mean = np.mean(actions, axis=0)
        a_std = np.std(actions, axis=0)
        print(f"[stat] action mean {a_mean}")
        print(f"[stat] action std {a_std}")
        drdas = np.array(self.replay_buffer.get_drda())
        drda_mean = np.mean(drdas, axis=0)
        drda_std = np.std(drdas, axis=0)
        print(f"[stat] drda mean {drda_mean}")
        print(f"[stat] drda std {drda_std}")

    def need_new_action(self):
        return self.world.env.need_new_action(self.id)

    def update(self, timestep):
        """update agent by a given timestep
        """
        if self.need_new_action():
            self._update_new_action()
        return

    def get_action_space(self):
        return self.world.env.get_action_space(self.id)

    def _check_action_space(self):
        action_space = self.get_action_space()
        return action_space == ActionSpace.Continuous

    def get_state_size(self):
        return self.world.env.get_state_size(self.id)

    def get_action_size(self):
        return self.world.env.get_action_size(self.id)

    def draw_param_hist(self):
        import matplotlib.pyplot as plt
        for _id, name in enumerate(self.action.state_dict()):
            # res = torch.histc().detach()
            plt.cla()
            param = np.array(self.action.state_dict()[name].detach())
            min_res = np.min(param)
            max_res = np.max(param)
            print(f"{name} min {min_res} max {max_res}")
            param = param.reshape(-1)
            plt.hist(param, bins=10)
            plt.title(f"{_id} {name}")
            filename = f"{_id} {name}.png"

            # plt.show()
            plt.savefig(filename)
            print(f"[log] save {filename}")

    def _build_graph(self, json_data):
        """
            Given the agent file, build the network from torch
        """
        assert self.POLICY_NET_KEY in json_data
        nets = json_data[self.POLICY_NET_KEY]
        assert self.ACTIVATION_KEY in json_data
        activation = json_data[self.ACTIVATION_KEY]
        assert self.DROPOUT_KEY in json_data
        dropout = json_data[self.DROPOUT_KEY]
        # print(f"dropout {dropout}")
        # exit(0)
        self.action = build_net(nets,
                                self.get_state_size(), self.get_action_size(), activation=activation, dropout=dropout)
        self.state_normalizer = NormalizerTorch(
            "state_normalizer", self.get_state_size(), self.world.env.build_state_norm_groups(self.id), self.normalizer_alpha)
        self.action_normalizer = NormalizerTorch(
            "action_normalizer", self.get_action_size(), alpha=self.normalizer_alpha)

    def _get_output_path(self):
        assert False
        return

    def _get_intermediate_output_path(self):
        assert False
        return

    def _record_state(self):
        s = self.world.env.record_state(self.id)
        return s

    def _is_first_step(self):
        return len(self.path.states) == 0

    def _record_reward(self):
        r = self.world.env.calc_reward(self.id)
        return r

    def _enable_draw(self):
        return self.world.env.enable_draw

    def log_reward(self, r):
        self.world.env.log_val(self.id, r)

    def _apply_action(self, a):
        self.world.env.set_action(self.id, a)
        return

    def _record_flags(self):
        return int(0)

    def _update_new_action(self):
        """
            1. inference a new action
            2. apply the new action to C++ core
            3. calc & record the state, action, reward, and some derivatives for diff Ctrl
        """
        # 1. record state
        s = self._record_state()

        assert np.isfinite(
            s).all() == True, f"some states is Nan!, s = {str(s)}"
        # 2. get the reward if it's not the first step
        if not (self._is_first_step()):
            r = self._record_reward()
            drda = self._record_drda()
            np.set_printoptions(suppress=False)
            # if self.enable_training == False and self._mode == self.Mode.TRAIN:
            print(
                f"[debug] action = {self.path.actions[-1]} drda = {drda} reward {r}")
            np.set_printoptions(suppress=True)
            np.set_printoptions(precision=5)
            # print(
            #     f"[debug] drda = {drda} reward {r}")
            self.path.rewards.append(r)
            self.path.drdas.append(drda)

            if self._enable_draw():
                self.log_reward(r)

                assert np.isfinite(
                    r).all() == True, f"some reward is Nan!, r = {str(r)}"

        a = self._decide_action(s, None)
        assert len(np.shape(a)) == 1, f"a shape {np.shape(a)}"
        print(f"[new action] {a}")
        flags = self._record_flags()
        assert np.isfinite(
            a).all() == True, f"some action is Nan!, a = {str(a)}"

        self._apply_action(a)
        self.path.states.append(s)
        self.path.actions.append(a)
        self.path.flags.append(flags)

    def _record_drda(self):
        drda = self.world.env.calc_drda(0)
        return drda

    def _end_path(self):
        """
            finished the current recorded path
            Note that a path != an episode. the latter one is broader
        """
        s = self._record_state()
        r = self._record_reward()
        drda = self._record_drda()
        self.path.rewards.append(r)
        self.path.states.append(s)
        self.path.drdas.append(drda)

        if self.enable_training == False:
            print(f"[test] path return {np.sum( self.path.rewards)}")
            print(f"[test] path avg rew {np.mean( self.path.rewards)}")
            print(f"[test] path avg drda {np.mean( self.path.drdas, axis =0)}")

        assert np.isfinite(s).all() == True and np.isfinite(r).all() == True

        self.path.terminate = self.world.env.check_terminate(self.id)
        return

    def get_enable_training(self):
        return self._enable_training

    def reset(self):
        self.path.clear()
        return

    def set_enable_training(self, enable):
        self._enable_training = enable
        if self._enable_training:
            self.reset()
        return

    enable_training = property(get_enable_training, set_enable_training)

    def end_episode(self):
        """
            the rl_world_torch will call this function when current episode ends
                1. time up
                2. falled down
            this function first end the path. if this path is valid, we begin to train

        """

        if self.path.pathlength() > 0:
            self._end_path()
            if self._mode == self.Mode.TRAIN or self._mode == self.Mode.TRAIN_END:
                if self.enable_training and self.path.pathlength() > 0:
                    self.replay_buffer.add(self.path)

                    # when the replay buffer is full
                    # if self.replay_buffer.get_cur_size() > self.replay_buffer.capacity:

                    # when the paths is up to 10
                    if self.replay_buffer.get_num_paths() > 20:
                        self._train()

            elif self._mode == self.Mode.TEST:
                self._update_test_return(self.path)
            else:
                assert False, print(
                    "Unsupported RL agent mode" + str(self._mode)
                )

            self._update_mode()
        return

    def _update_mode(self):
        pre_mode = self._mode
        if self._mode == self.Mode.TRAIN:
            self._update_mode_train()
        elif self._mode == self.Mode.TRAIN_END:
            self._update_mode_train_end()
        elif self._mode == self.Mode.TEST:
            self._update_mode_test()
        else:
            assert False, f"Unsupported agent mode {str(self._mode)}"
        if pre_mode != self._mode:
            print(f"[mode] mode update from {pre_mode} to {self._mode}")

    def _update_mode_train(self):
        return

    def _update_mode_train_end(self):
        print("[debug] convert mode from train end to test")
        self._mode = self.Mode.TEST
        self.test_return = 0.0
        self.test_episode_count = 0
        self.world.env.set_mode(self._mode)

    def _update_mode_test(self):
        # if we have trained for enough episodes
        if self.test_episode_count * MPIUtil.get_num_procs() >= self.test_episodes:
            print("[debug] convert mode from test to train")
            if self.enable_training:
                # print(
                #     f"current test episode {self.test_episode_count} * proc_num {MPIUtil.get_num_procs()} >= {self.test_episodes}, convert to train mode")
                self._mode = self.Mode.TRAIN
                self.world.env.set_mode(self._mode)

            self.test_return /= self.test_episode_count
            print(f"[test] test return = {self.test_return}")
            self.test_episode_count = 0
            self.test_return = 0

    def _update_test_return(self, path):
        path_return = np.sum(path.rewards)

        self.test_return += path_return
        self.test_episode_count += 1
        return

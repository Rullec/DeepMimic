import numpy as np
import copy
import os
import time
import json
import datetime
from env.action_space import ActionSpace
from abc import ABC, abstractmethod
from enum import Enum
from learning.tf.rl_agent import RLAgent
from learning.torch.nets.net_builder import build_net
import torch
import util.mpi_util as MPIUtil
import torch.optim as optim
from learning.torch.replay_buffer_torch import ReplayBufferTorch


class TorchAgent(RLAgent):
    """
        Torch Agent
    """
    NAME = "DiffMBRL"
    POLICY_NET_KEY = "PolicyNet"
    POLICY_STEPSIZE_KEY = "PolicyStepsize"
    POLICY_MOMENTUM_KEY = "PolicyMomentum"
    POLICY_WEIGHT_LOSS_KEY = "PolicyWeightLoss"

    def __init__(self, world, id, json_data):
        """
            Init method, create from json data
        """
        super().__init__(world, id, json_data)
        self._build_graph(json_data)

        self.optimizer = optim.SGD(
            self.action.parameters(), lr=3e-5)
        self.replay_buffer = ReplayBufferTorch(self.replay_buffer_size)
        return

    def save_model(self, out_path):
        tar_dir = os.path.dirname(out_path)
        if False == os.path.exists(tar_dir):
            os.makedirs(tar_dir)
        torch.save(self.action.state_dict(), out_path)

        print(
            f"[log] torch save model to {out_path}, param {np.sum([np.linalg.norm(i.detach()) for i in self.action.parameters()])}")

        return

    def load_model(self, in_path):
        if os.path.exists(in_path) == False:
            return
        self.action.load_state_dict(torch.load(in_path))
        print(
            f"[log] torch load model from {in_path}, param {np.sum([np.linalg.norm(i.detach()) for i in self.action.parameters()])}")
        # exit(0)
        return

    def _decide_action(self, s, g):
        a = self.action(torch.Tensor(s)).detach().numpy()
        # print(f"[debug] decide new action {a}")
        return a

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

    def _train(self):
        """
            this function is called when the update_counters >= update_period
            train the current agent
        """
        # 1. begin to train, grac = dr/da * da/d\theta
        # path_len = self.path.get_pathlen()
        samples = self.replay_buffer.get_size()
        x = np.array(self.replay_buffer.get_state())
        w = np.array(self.replay_buffer.get_drda())
        # print(f"states {states}")
        # print(f"states type {type(states)}")
        # print(f"states {states.shape}")
        print(
            f"[train] states shape {x.shape}, drdas shape {w.shape}, lr {self._get_lr()}")

        y_torch = self.action(torch.Tensor(x))
        w_torch = torch.Tensor(w)
        assert(w_torch.shape == y_torch.shape)
        pesudo_loss = y_torch * w_torch  # y: [batch, m], w:[batch, m]
        assert(pesudo_loss.shape == w_torch.shape)
        persudo_loss_sum = -torch.sum(pesudo_loss)
        persudo_loss_sum.backward()

        self.optimizer.step()
        self._set_lr(max(0.98 * self._get_lr(), 1e-7))

        # 2. update sample count, update time params
        self._total_sample_count += samples
        self.world.env.set_sample_count(
            self._total_sample_count + self.beginning_sample_count)
        print(
            f"beginning {self.beginning_sample_count}, total {self._total_sample_count}")
        self._update_exp_params()

        # 3. output and clear
        output_name = datetime.datetime.now().strftime("%m-%d-%H:%M:%S")
        output_name = f"{output_name}-{str(self.replay_buffer.get_avg_reward())[:5]}.pkl"
        output_path = os.path.join(self.output_dir, output_name)
        self.save_model(output_path)
        self.replay_buffer.clear()

    def update(self, timestep):
        """update agent by a given timestep
        """
        if self.need_new_action():
            self._update_new_action()
        return

    def _check_action_space(self):
        action_space = self.get_action_space()
        return action_space == ActionSpace.Continuous

    def _build_graph(self, json_data_path):
        """
            Given the agent file, build the network from torch
        """
        with open(json_data_path) as f:
            json_data = json.load(f)
        assert self.POLICY_NET_KEY in json_data
        assert self.POLICY_STEPSIZE_KEY in json_data
        assert self.POLICY_MOMENTUM_KEY in json_data
        assert self.POLICY_WEIGHT_LOSS_KEY in json_data

        self.action = build_net(json_data[self.POLICY_NET_KEY],
                                self.get_state_size(), self.get_action_size())

    def _get_output_path(self):
        assert False
        return

    def _get_intermediate_output_path(self):
        assert False
        return

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
            # print(f"[debug] drda = {drda}, reward {r}")
            self.path.rewards.append(r)
            self.path.drdas.append(drda)

            if self._enable_draw():
                self.log_reward(r)

                assert np.isfinite(
                    r).all() == True, f"some reward is Nan!, r = {str(r)}"

        a = self._decide_action(s, None)
        assert len(np.shape(a)) == 1, f"a shape {np.shape(a)}"

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
        g = self._record_goal()
        r = self._record_reward()
        drda = self._record_drda()
        self.path.rewards.append(r)
        self.path.states.append(s)
        self.path.drdas.append(drda)
        # print(f"[debug] torch path end, drdas {self.path.drdas}")

        assert np.isfinite(s).all() == True and np.isfinite(r).all() == True

        self.path.goals.append(g)
        self.path.terminate = self.world.env.check_terminate(self.id)
        return

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

                    if self.replay_buffer.get_size() > 200:
                        self._train()

            elif self._mode == self.Mode.TEST:
                self._update_test_return(self.path)
            else:
                assert False, print(
                    "Unsupported RL agent mode" + str(self._mode)
                )

            self._update_mode()
        return

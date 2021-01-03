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
        return

    def save_model(self, out_path):
        assert False, "save model hasn't been implemented"
        return

    def load_model(self, in_path):
        assert False, "load model hasn't been implemented"
        return

    def _decide_action(self, s, g):
        a = self.action(torch.Tensor(s)).detach().numpy()
        print(f"[debug] new action {a}")
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
            self.path.rewards.append(r)

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

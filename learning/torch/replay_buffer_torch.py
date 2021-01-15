import numpy as np
import os
import datetime
from learning.torch.path_torch import PathTorch


class ReplayBufferTorch(object):
    """
        Replay buffer, used to store paths
    """

    def __init__(self, buffer_size):
        self.capacity = buffer_size
        self.clear()

    def add(self, new_path: PathTorch):
        assert type(new_path) == PathTorch
        self.state_lst += new_path.states[:-1]
        self.drda_lst += new_path.drdas
        self.reward_lst += new_path.rewards
        self.action_lst += new_path.actions

    def get_action(self):
        return self.action_lst

    def get_state(self):
        return self.state_lst

    def get_cur_size(self):
        return len(self.state_lst)

    def get_reward(self):
        return self.reward_lst

    def get_drda(self):
        return self.drda_lst

    def get_avg_reward(self):
        return self.get_total_reward() / self.get_cur_size()

    def get_total_reward(self):
        return np.sum(self.reward_lst)

    def clear(self):
        self.state_lst = []
        self.drda_lst = []
        self.reward_lst = []
        self.action_lst = []

from typing import Optional, List, Any, Union, Tuple

from learning.tf.tf_agent import TFAgent
import learning.tf.tf_util as TFUtil
from util.io import load_buffer_npz
import numpy as np
import tensorflow as tf
# from retargeting.sa.SimpleAgentNetWork import NetWork
from retargeting.sa.SimpleAgentNetWorkOpt import NetWork


class SimpleAgent(TFAgent):
    NAME = 'SA'
    STATE_MEAN_PATH = 'state_mean_path'
    STATE_STD_PATH = 'state_std_path'
    ACTION_MEAN_PATH = 'action_mean_path'
    ACTION_STD_PATH = 'action_std_path'
    ENABLE_ACTION_NORMALIZE = 'enable_action_normalize'

    def __init__(self, world, id, json_data):
        self.network_initialized = False
        self.state_mean_path = json_data[self.STATE_MEAN_PATH]
        self.state_std_path = json_data[self.STATE_STD_PATH]
        self.action_mean_path = json_data[self.ACTION_MEAN_PATH]
        self.action_std_path = json_data[self.ACTION_STD_PATH]
        self.enable_action_normalize = json_data[self.ENABLE_ACTION_NORMALIZE]
        self.init_agent(json_data)

        super().__init__(world, id, json_data)

    def _build_normalizers(self):
        self.state_std = np.load(self.state_std_path)['std']
        self.state_mean = np.load(self.state_mean_path)['mean']

        if True == self.enable_action_normalize:
            self.action_std = np.load(self.action_std_path)['std']
            self.action_mean = np.load(self.action_mean_path)['mean']

    def _norm_s(self, s):
        return (s - self.state_mean) / self.state_std

    def _unnorm_a(self, a):
        # print("unnormalize!\n")
        return a * self.action_std + self.action_mean

    def _build_bounds(self):
        if self.world is not None:
            super()._build_bounds()

    def _build_losses(self, json_data):
        return

    def _build_solvers(self, json_data):
        pass

    def _decide_action(self, s, g):
        s = np.reshape(s, [1, len(s)])
        a_0, logps, a_mean = self.net.eval(self._norm_s(s))
        if self.enable_action_normalize == True:
            return self._unnorm_a(a_0), logps, self._unnorm_a(a_mean)
        else:
            return a_0, logps, a_mean

    def _train_step(self):
        pass

    def _check_action_space(self):
        return True

    def _build_nets(self, json_data):
        s_size = self.get_state_size()
        a_size = self.get_action_size()
        self.net = NetWork(s_size, a_size, json_data=json_data)

    def init_agent(self, json_data):
        pass

    def _build_saver(self):
        self.saver = tf.train.Saver()

    def _init_normalizers(self):
        if self.network_initialized:
            super()._init_normalizers()

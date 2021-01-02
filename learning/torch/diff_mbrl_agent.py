import numpy as np
import copy
import os
import time
import datetime

from abc import ABC, abstractmethod
from enum import Enum
from learning.tf.rl_agent import RLAgent


class DiffMBRLAgent(RLAgent):
    """
        Model-based RL agent for differential control
    """
    NAME = "DiffMBRL"

    def __init__(self, world, id, json_data):
        """
            Init method, create from json data
        """
        super().__init__(world, id, json_data)
        return

    def save_model(self, out_path):
        assert False
        return

    def load_model(self, in_path):
        assert False
        return

    def _decide_action(self, s, g):
        assert False
        return

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
        assert False
        return

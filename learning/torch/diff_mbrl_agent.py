import numpy as np
import copy
import os
import time
import datetime

from abc import ABC, abstractmethod
from enum import Enum
from learning.torch.torch_agent import TorchAgent
# from torch_agent import TorchAgent

class DiffMBRLAgent(TorchAgent):
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

    def _train_step(self):
        assert False
        return
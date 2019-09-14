import numpy as np
import copy
from util.logger import Logger
import inspect as inspect
from env.env import Env
import util.math_util as MathUtil

class ReplayBuffer_xudong(object):
    TERMINATE_KEY = 'terminate'
    PATH_START_KEY = 'path_start'
    PATH_END_KEY = 'path_end'

    def __init__(self, buffer_size):
        assert buffer_size > 0
        self.buffer_size = buffer_size
        self.total_count = 0
        self.buffer_head = 0
        self.buffer_tail = MathUtil.INVALID_IDX
        self.num_paths = 0
        self.buffers = None

        self.clear()

    def clear(self):
        self.buffer_head = 0
        self.buffer_tail = MathUtil.INVALID_IDX
        self.num_paths = 0
        
        return

    def sample(self, n):
        return None

    def get(self, key, idx):
        return None

    def get_all(self, key, idx):
        return None

    def get_path_end(self, idx):
        return None
    
    def get_current_size(self):
        return None
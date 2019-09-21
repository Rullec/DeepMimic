import numpy as np
from env.env import Env

class Path(object):
    '''
        存储当前采样path的trajectory，即[s, a, r, logp]键值对
    '''
    def __init__(self):
        self.clear()
        return

    def pathlength(self):
        return len(self.actions)

    def is_valid(self):
        valid = True
        l = self.pathlength()
        valid &= len(self.states) == l + 1
        valid &= len(self.actions) == l
        valid &= len(self.logps) == l
        valid &= len(self.rewards) == l

        return valid

    def check_vals(self):
        for i_out, vals in enumerate([self.states, self.actions, self.logps,
                  self.rewards]):
            for i_in, v in enumerate(vals):
                if not np.isfinite(v).all():
                    print("i_out: " + str(i_out))
                    print("i_in: %d / %d" % (i_in, len(vals)) )
                    assert 0 == 1
                    return False
        return True

    def clear(self):
        self.states = []
        self.actions = []
        self.logps = []
        self.rewards = []
        self.flags = []
        self.terminate = Env.Terminate.Null
        return

    def get_pathlen(self):
        return len(self.rewards)

    def calc_return(self):
        return sum(self.rewards)
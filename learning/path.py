import os
import json
import numpy as np
from env.env import Env

# 这个类是干什么用的? 似乎是检测trajectory有效性的?
class Path(object):
    def __init__(self):
        self.clear()
        return

    def pathlength(self):
        return len(self.actions)

    def is_valid(self):
        valid = True
        l = self.pathlength()
        valid &= len(self.states) == l + 1
        valid &= len(self.goals) == l + 1
        valid &= len(self.actions) == l
        valid &= len(self.logps) == l
        valid &= len(self.rewards) == l
        valid &= len(self.flags) == l

        return valid

    def check_vals(self):
        for i_out, vals in enumerate([self.states, self.goals, self.actions, self.logps,
                  self.rewards]):
            for i_in, v in enumerate(vals):
                if not np.isfinite(v).all():
                    print("i_out:" + str(i_out))
                    print("i_in: %d / %d" % (i_in, len(vals)) )
                    assert 0 == 1
                    return False
        return True

    def clear(self):
        self.states = []
        self.poses = []
        self.goals = []
        self.actions = []
        self.logps = []
        self.rewards = []
        self.flags = []
        self.contact_info = []
        self.terminate = Env.Terminate.Null
        return

    def get_pathlen(self):
        return len(self.rewards)

    def calc_return(self):
        return sum(self.rewards)

    def save(self, filename):
        pass
        if os.path.exists(filename):
            os.remove(filename)
        cont = {
            "states" : [i.tolist() for i in self.states],
            "poses" : [i.tolist() for i in self.poses],
            "actions" : [i.tolist() for i in self.actions],
            "logps" : [float(i) for i in self.logps],
            "rewards" : [ float(i) for i in self.rewards],
            "contact_info" : [i.tolist() for i in self.contact_info]
        }
        with open(filename, "w") as f:
            json.dump(cont, f, indent=4)

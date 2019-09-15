from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from enum import Enum

'''
    class Agent_xudong is a abstract base class, which offer some very basic interfaces for agents in Reinforcement learning.

'''
class Agent_xudong(ABC):
    class Mode(Enum):
        TRAIN = 0
        TEST = 1
        TRAIN_END = 2

    # keys name
    AGENTTYPE_KEYS = "AgentType"

    def __init__(self, world, id):
        self.world = world
        self.id = id
        self.state_size = self.world.env.get_state_size(self.id)
        self.action_size = self.world.env.get_action_size(self.id)
        self.action_lower_bound = self.world.env.build_action_bound_min(self.id)
        self.action_upper_bound = self.world.env.build_action_bound_max(self.id)
        self._mode = self.Mode.TRAIN
        
        # the tf.sess created
        self.graph = tf.Graph()
        self.sess = tf.Session(graph = self.graph)
        
        print("[agent_xudong] initialized, id {}".format(id))

    @abstractmethod
    def decide_action(self, state):
        pass

    @abstractmethod
    def load_params(self, json_data):
        pass

    def get_action_space(self):
        return self.world.env.get_action_space(self.id)

    def get_num_actions(self):
        return self.world.env.get_num_actions(self.id)
    
    def need_new_action(self):
        """Judge whether an agent need a new action

            To illustrate this function, Let's take an overview about the whole simulation process.
            There are 2 frequencies in this context, basically. First I call it "simulation frenquency", which means 
            that the simulation enviroment would update itself,  100 times per second(even higher). 
            And the second, "action request frequency" for the controller to give an explicit action
            based on the current "state" or so-called "observation", 30 times per second.

            So when simulation env needs to update itself after 1/100s, usually the agent don't need to give an action but let it run arbitarily.
            how can we judge it? just call this func "need_new_action" and make the simulation env tells you whether the agent needs to give an action...
        """
        return self.world.env.need_new_action(self.id)
    
    def enable_draw(self):
        return self.world.env.enable_draw
    
    def tf_vars(self, scope=''):
        with self.sess.as_default(), self.graph.as_default():
            res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            assert len(res) > 0
        return res
    
    def _calc_term_vals(self, discount):
        r_fail = self.world.env.get_reward_fail(self.id)
        r_succ = self.world.env.get_reward_succ(self.id)

        r_min = self.world.env.get_reward_min(self.id)
        r_max = self.world.env.get_reward_max(self.id)
        assert(r_fail <= r_max and r_fail >= r_min)
        assert(r_succ <= r_max and r_succ >= r_min)
        assert(not np.isinf(r_fail))
        assert(not np.isinf(r_succ))

        if (discount == 0):
            val_fail = 0
            val_succ = 0
        else:
            val_fail = r_fail / (1.0 - discount)
            val_succ = r_succ / (1.0 - discount)

        return val_fail, val_succ

    def _calc_val_bounds(self, discount):
        r_min = self.world.env.get_reward_min(self.id)
        r_max = self.world.env.get_reward_max(self.id)
        assert(r_min <= r_max)

        val_min = r_min / ( 1.0 - discount)
        val_max = r_max / ( 1.0 - discount)
        return val_min, val_max
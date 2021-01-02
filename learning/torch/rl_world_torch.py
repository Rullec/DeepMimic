import numpy as np
import learning.torch.agent_builder_torch as AgentBuilderTorch


class RLWorldTorch(object):
    """Reinforcement World in torch
    """

    def __init__(self, env, arg_parser):
        self.env = env
        self.arg_parser = arg_parser

        self._enable_training = True
        self.train_agents = []
        self.parse_args(arg_parser)
        self.build_agents()

        return

    def parse_args(self, sarg_parser):
        self.train_agents = self.arg_parser.parse_bools('train_agents')
        num_agents = self.env.get_num_agents()

        assert(len(self.train_agents) ==
               num_agents or len(self.train_agents) == 0)

        return

    def build_agents(self):
        assert False
        return

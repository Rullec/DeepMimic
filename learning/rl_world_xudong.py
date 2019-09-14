import numpy as np
import learning.agent_builder_xudong as AgentBuilder
import learning.tf_util as TFUtil
from learning.agent_xudong import Agent_xudong
from util.logger import Logger

class RLWorld_xudong(object):
    def __init__(self, env, arg_parser):
        TFUtil.disable_gpu()

        self.env = env              # simulation环境是进到RLworld里来的。
        self.arg_parser = arg_parser
        self._enable_training = True
        self.train_agents = []
        self.parse_args(arg_parser) # 解析参数，主要解释num_agent, agent的个数。

        self.build_agents()
        
        return

    def set_enable_training(self, enable):
        self._enable_training = enable
        for i in range(len(self.agents)):
            curr_agent = self.agents[i]
            if curr_agent is not None:
                enable_curr_train = self.train_agents[i] if (len(self.train_agents) > 0) else True
                curr_agent.enable_training = self.enable_training and enable_curr_train

        if (self._enable_training):
            self.env.set_mode(Agent_xudong.Mode.TRAIN)
        else:
            self.env.set_mode(Agent_xudong.Mode.TEST)

        return

    def get_enable_training(self):
        return self._enable_training

    enable_training = property(get_enable_training, set_enable_training)

    def build_agents(self):
        num_agents = self.env.get_num_agents()
        self.agents = []

        Logger.print('')
        Logger.print('Num Agents: {:d}'.format(num_agents))

        # 在RL world中，拿到agent文件、模型文件、输出路径
        # agent包括:
        agent_files = self.arg_parser.parse_strings('agent_files')
        assert(len(agent_files) == num_agents or len(agent_files) == 0)

        # model: .ckpt checkpoint
        model_files = self.arg_parser.parse_strings('model_files')
        assert(len(model_files) == num_agents or len(model_files) == 0)

        # model的输出path
        output_path = self.arg_parser.parse_string('output_path')

        # agent只有一个, worker可以有很多个。多个worker只是用来充分利用资源扩大采样而已。
        for i in range(num_agents):
            curr_file = agent_files[i]

            # build agent 就在这行，对agent建立actor critic网络
            curr_agent = self._build_agent(i, curr_file)

            if curr_agent is not None:
                curr_agent.output_dir = output_path
                Logger.print(str(curr_agent))

                if (len(model_files) > 0):
                    curr_model_file = model_files[i]
                    if curr_model_file != 'none':
                        curr_agent.load_model(curr_model_file)
                        curr_agent.save_model(curr_model_file)

            self.agents.append(curr_agent)
            Logger.print('')

        self.set_enable_training(self.enable_training)

        return

    def parse_args(self, arg_parser):
        self.train_agents = self.arg_parser.parse_bools('train_agents')
        num_agents = self.env.get_num_agents()
        # agent 个数和 worker个数不一样，一个agent代表训练一个策略;而多个worker只是用来增加采样数的
        assert(len(self.train_agents) == num_agents or len(self.train_agents) == 0)

        return

    def _build_agent(self, id, agent_file):
        Logger.print('Agent {:d}: {}'.format(id, agent_file))
        if (agent_file == 'none'):
            agent = None
        else:
            agent = AgentBuilder.build_agent(self, id, agent_file)
            assert (agent != None), 'Failed to build agent {:d} from: {}'.format(id, agent_file)
        
        return agent
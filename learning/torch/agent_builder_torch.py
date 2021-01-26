import json
import numpy as np

AGENT_TYPE_KEY = "AgentType"
from learning.torch.diff_mbrl_agent import DiffMBRLAgent
from learning.torch.torch_inde_agent import TorchIndeAgent


def build_agent_torch(world, id, file):
    """
        Create the new agent from a file
    """
    agent = None
    with open(file) as data_file:
        json_data = json.load(data_file)

        assert AGENT_TYPE_KEY in json_data

        agent_type = json_data[AGENT_TYPE_KEY]
        if agent_type == DiffMBRLAgent.NAME:
            agent = DiffMBRLAgent(world, id, json_data)
        elif agent_type == TorchIndeAgent.NAME:
            agent = TorchIndeAgent(world, id, json_data)
        else:
            assert False, f"invalid agent type {agent_type}"
    return agent

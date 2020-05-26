import os
import logging
import json
import numpy as np
from pathlib import Path

root_dir = ""
old_format_data = [str(i) for i in sorted(Path("/home/xudong/Projects/DeepMimic/data/paths").iterdir(), key = os.path.getmtime)]
new_format_data = [str(i) for i in sorted(Path("/home/xudong/Projects/DeepMimic/data/batch_train_data/0526").iterdir(), key = os.path.getmtime)]
assert(len(old_format_data) == len(new_format_data))
# root_dir = os.getcwd()
# new_format_data = ["data/batch_train_data/0526/traj_fullbody_1620270081.train", "data/batch_train_data/0526/traj_fullbody_602155232.train"]
# old_format_data = ["data/paths/2020-05-26_12-19-44.150084.json", "data/paths/2020-05-26_12-19-41.550105.json"]

logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
formatter = logging.Formatter("%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
fh = logging.FileHandler("data_diff.txt", 'w')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

def parse_old_format_data(filename):
    global logger
    assert(os.path.exists(filename))
    logger.debug("loading old format %s" % filename)
    with open(filename, 'r') as f:
        root = json.load(f)  
    # print(root.keys())
    states_json = root["states"]
    actions_json = root["actions"]
    rewards_json = root["rewards"]
    num_of_frames = len(states_json) - 1
    assert(num_of_frames == len(actions_json))
    assert(num_of_frames == len(rewards_json))

    states = []
    actions = []
    rewards = []
    for i in range(num_of_frames):
        states.append([j for j in states_json[i]])
        actions.append([j for j in actions_json[i]])
        assert(type(rewards_json[i]) is float)
        rewards.append(rewards_json[i])
    return num_of_frames, states, actions, rewards

def parse_new_format_data(filename, sample_num):
    global logger
    assert(os.path.exists(filename))
    logger.debug("loading new format %s" % filename)
    with open(filename, 'r') as f:
        root = json.load(f)
    num_of_frames = root["num_of_frames"] // sample_num
    data_lst = root["data_list"]
    states = []
    rewards = []
    actions = []
    for id, single_frame in enumerate(data_lst):
        if id % sample_num == 0:
            actions.append([i for i in single_frame["action"]])
            states.append([i for i in single_frame["state"]])
            assert(type(single_frame["reward"]) is float)
            rewards.append(single_frame["reward"])
    
    assert(num_of_frames == len(actions))
    assert(num_of_frames == len(rewards))
    assert(num_of_frames == len(states))
    return num_of_frames, states, actions, rewards

if __name__ == "__main__":
    for data_id in range(len(new_format_data)):
        # 1. load buffers
        old_nums, old_states, old_actions, old_rewards = parse_old_format_data(os.path.join(root_dir, old_format_data[data_id]))
        new_nums, new_states, new_actions, new_rewards = parse_new_format_data(os.path.join(root_dir, new_format_data[data_id]), sample_num = 20)
        logger.info("---------------------data id %d--------------------", data_id)
        # 2. begin to diff them
        logger.debug("old_format & new_format info")
        logger.debug("num_of_frames: %d - %d" % (old_nums, new_nums))
        assert(old_nums == new_nums)

        state_diff = 0
        action_diff = 0
        for i in range(old_nums):
            logger.info("for frame %d state: %lf - %lf", i, np.linalg.norm(np.array(old_states[i])), np.linalg.norm(np.array(new_states[i])))
            logger.info("for frame %d action: %lf - %lf", i, np.linalg.norm(np.array(old_actions[i])), np.linalg.norm(np.array(new_actions[i])))
            logger.info("for frame %d reward: %lf - %lf", i, np.linalg.norm(np.array(old_rewards[i])), np.linalg.norm(np.array(new_rewards[i])))
            logger.debug("new state %s", str(new_states[i]))
            logger.debug("old state %s", str(old_states[i]))
            logger.debug("new action %s", str(new_actions[i]))
            logger.debug("old action %s", str(old_actions[i]))
            if i != 0:
                state_diff += np.linalg.norm(np.array(old_states[i]) - np.array(new_states[i]))
                action_diff += np.linalg.norm(np.array(old_actions[i]) - np.array(new_actions[i]))
            logger.info("-----------------------------------")
        logger.info("for traj %d, state diff = %.3f", data_id, state_diff)
        logger.info("for traj %d, action diff = %.3f", data_id, action_diff)

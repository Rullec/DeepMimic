"""
MR = Motion retargeting
Given a batch of train data [state, action] before MR and after MR, show the mean and std for each bit of [state, action] and compare them
"""
import os
import numpy as np
import json
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger("state_action_statistic")
logger.setLevel(logging.DEBUG)
a = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
a.setFormatter(formatter)
logger.addHandler(a)

cwd = "../"
old_data_summary_table_path = (
    "data/batch_train_data/0609_sampled_solved/summary_table_fullbody.json"
)
new_data_summary_table_path = (
    "data/batch_train_data/0609_mred_solved/same_trans_table.json"
)

split_intervals_num = 1000
def get_state_action_staticstics_by_phase(table_path: str):
    assert os.path.exists(table_path), f"{table_path} doesn't exist"
    with open(table_path, "r") as f:
        root_json = json.load(f)

    ID_traindata_dir = os.path.join(cwd, root_json["ID_traindata_dir"])

    assert os.path.exists(ID_traindata_dir), f"{ID_traindata_dir} doesn't exist"
    state_npz_path = os.path.join(ID_traindata_dir, "states.npz")
    action_npz_path = os.path.join(ID_traindata_dir, "actions.npz")

    assert os.path.exists(state_npz_path), f"{state_npz_path} doesn't exist"
    assert os.path.exists(action_npz_path), f"{action_npz_path} doesn't exist"

    state_data = dict(np.load(state_npz_path, "r"))["s"]
    action_data = dict(np.load(action_npz_path, "r"))["a"]

    logger.info(f"load state data from {state_npz_path} succ, nums {state_data.shape}")
    
    logger.info(
        f"load action data from {action_npz_path} succ, nums {action_data.shape}"
    )

    state_lst = [[] for i in range(split_intervals_num)]
    action_lst = [[] for i in range(split_intervals_num)]
    total_num_frames = state_data.shape[0]
    print(f"frame {total_num_frames}" )
    for i in range(total_num_frames):
        phase = state_data[i, 0]
        phase_int = int(phase * split_intervals_num)
        # print(f"phase {phase_int}")
        state_lst[phase_int].append(state_data[i])
        action_lst[phase_int].append(action_data[i])
    
    state_mean_lst = [0 for i in range(split_intervals_num)]
    state_std_lst = [0 for i in range(split_intervals_num)]
    action_mean_lst = [0 for i in range(split_intervals_num)]
    action_std_lst = [0 for i in range(split_intervals_num)]
    for i in range(split_intervals_num):
        state_norm_this_phase = np.linalg.norm(np.array(state_lst[i]), axis=1)
        state_mean_lst[i] = np.mean(state_norm_this_phase)
        state_std_lst[i] = np.std(state_norm_this_phase)

        action_norm_this_phase = np.linalg.norm(np.array(action_lst[i]), axis=1)
        action_mean_lst[i] = np.mean(action_norm_this_phase)
        action_std_lst[i] = np.std(action_norm_this_phase)
        # print(f"{i} shape {action_norm_this_phase.shape}")
        # state_lst[i] = np.linalg.norm(np.array(state_lst[i]))
    return state_mean_lst, state_std_lst, action_mean_lst, action_std_lst
    # print(f"state mean list {state_mean_lst}")

def get_state_action_staticstics_roughly(table_path: str):
    assert os.path.exists(table_path), f"{table_path} doesn't exist"

    with open(table_path, "r") as f:
        root_json = json.load(f)

    ID_traindata_dir = os.path.join(cwd, root_json["ID_traindata_dir"])

    assert os.path.exists(ID_traindata_dir), f"{ID_traindata_dir} doesn't exist"
    state_npz_path = os.path.join(ID_traindata_dir, "states.npz")
    action_npz_path = os.path.join(ID_traindata_dir, "actions.npz")

    assert os.path.exists(state_npz_path), f"{state_npz_path} doesn't exist"
    assert os.path.exists(action_npz_path), f"{action_npz_path} doesn't exist"

    state_data = dict(np.load(state_npz_path, "r"))["s"]
    action_data = dict(np.load(action_npz_path, "r"))["a"]

    logger.info(f"load state data from {state_npz_path} succ, nums {state_data.shape}")
    logger.info(
        f"load action data from {action_npz_path} succ, nums {action_data.shape}"
    )
    state_std = np.std(state_data, axis=0)
    state_mean = np.mean(state_data, axis=0)
    action_std = np.std(action_data, axis=0)
    action_mean = np.mean(action_data, axis=0)

    return state_mean, state_std, action_mean, action_std
    # print(state_std)
    # print(state_mean)
    # plt.plot(state_std)
    # plt.plot(state_mean)
    # plt.xlabel("state_id")
    # plt.legend(["state_std", "state_mean"])
    # plt.ylabel("val")
    # # for i in range(len(state_std)):
    #     # plt.text(i, state_std, str(i))
    #     # plt.text(state_mean)
    # plt.show()

def subplot_wrapper(old_value, new_value, title, id):
    plt.subplot(2, 2, id)
    plt.title(title)
    plt.plot(old_value)
    plt.plot(new_value)
    plt.legend(["old", "new"])
    
if __name__ == "__main__":

    # old_s_std, old_s_mean, old_a_std, old_a_mean = get_state_action_staticstics( os.path.join(cwd, old_data_summary_table_path))
    # new_s_std, new_s_mean, new_a_std, new_a_mean = get_state_action_staticstics(
    #     os.path.join(cwd, new_data_summary_table_path)
    # )

    # draw 4 figures. for state/action x mean/stde
    # plt.subplot(221)
    # plt.title("state std")
    # plt.xlabel("pos_id")
    # plt.plot(old_s_std, 'b-.')
    # plt.plot(new_s_std, 'r-.')
    # plt.legend(["old", "new"])

    # plt.subplot(222)
    # plt.title("state mean")
    # plt.plot(old_s_mean, 'b-.')
    # plt.plot(new_s_mean, 'r-.')
    # plt.xlabel("pos_id")
    # print("new state mean shape ", new_s_mean.shape)
    # print("old state mean shape", old_s_mean.shape)
    # for i in range(50):
    #     print(f"state mean {i} new {new_s_mean[i]} old {old_s_mean[i]}")
    # plt.legend(["old", "new"])

    # plt.subplot(223)
    # plt.title("action std")
    # plt.xlabel("pos_id")
    # plt.plot(old_a_std, 'b-.')
    # plt.plot(new_a_std, 'r-.')
    # plt.legend(["old", "new"])

    # plt.subplot(224)
    # plt.title("action mean")
    # plt.xlabel("pos_id")
    # plt.plot(old_a_mean, 'b-.')
    # plt.plot(new_a_mean, 'r-.')
    # plt.legend(["old", "new"])
    # plt.show()

    new_s_mean, new_s_std, new_a_mean, new_a_std = get_state_action_staticstics_by_phase( os.path.join(cwd, new_data_summary_table_path))
    old_s_mean, old_s_std, old_a_mean, old_a_std = get_state_action_staticstics_by_phase( os.path.join(cwd, old_data_summary_table_path))

    subplot_wrapper(old_s_mean, new_s_mean, "state mean", 1)
    subplot_wrapper(old_s_std, new_s_std, "state std", 2)
    subplot_wrapper(old_a_mean, new_a_mean, "action mean", 3)
    subplot_wrapper(old_a_std, new_a_std, "aciton std", 4)
    plt.show()
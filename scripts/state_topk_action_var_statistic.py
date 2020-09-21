"""
Given a batch of train data [state, action], clustering
"""
import os
import sys
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
import sklearn.cluster as sc
import torch

logger = logging.getLogger("state_action_statistic")
logger.setLevel(logging.DEBUG)
a = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
a.setFormatter(formatter)
logger.addHandler(a)

cwd = "../"
# get the state and action of paths
def get_paths(data_dir: str):
    assert os.path.exists(data_dir), f"{data_dir} doesn't exist"
    state_npz_path = os.path.join(data_dir, "states.npz")
    action_npz_path = os.path.join(data_dir, "actions.npz")

    assert os.path.exists(state_npz_path), f"{state_npz_path} doesn't exist"
    assert os.path.exists(action_npz_path), f"{action_npz_path} doesn't exist"

    state_data = dict(np.load(state_npz_path, "r"))["s"].astype(np.float32)
    action_data = dict(np.load(action_npz_path, "r"))["a"].astype(np.float32)

    logger.info(f"load state data from {state_npz_path} succ, nums {state_data.shape}")

    logger.info(
        f"load action data from {action_npz_path} succ, nums {action_data.shape}"
    )
    return state_data, action_data


def get_train_data(table_path: str):
    assert os.path.exists(table_path), f"{table_path} doesn't exist"
    with open(table_path, "r") as f:
        root_json = json.load(f)

    ID_traindata_dir = os.path.join(cwd, root_json["ID_traindata_dir"])

    assert os.path.exists(ID_traindata_dir), f"{ID_traindata_dir} doesn't exist"
    state_npz_path = os.path.join(ID_traindata_dir, "states.npz")
    action_npz_path = os.path.join(ID_traindata_dir, "actions.npz")

    assert os.path.exists(state_npz_path), f"{state_npz_path} doesn't exist"
    assert os.path.exists(action_npz_path), f"{action_npz_path} doesn't exist"

    state_data = np.array(dict(np.load(state_npz_path, "r"))["s"], dtype=np.float32)
    action_data = np.array(dict(np.load(action_npz_path, "r"))["a"], dtype=np.float32)

    logger.info(f"load state data from {state_npz_path} succ, nums {state_data.shape}")

    logger.info(
        f"load action data from {action_npz_path} succ, nums {action_data.shape}"
    )
    return state_data, action_data


traj_summary_path = (
    "/home/xudong/Projects/DeepMimic/data/id_test/solved_legs_short/summary_legs.json"
)
path_dir_path = "/home/xudong/Projects/DeepMimic/data/path_legs_short"



# 1. load all of the train data
# traj_state, traj_action = get_train_data(traj_summary_path)
# path_state, path_action = get_paths(path_dir_path)

# 2. calculate the eucleadian distance between each elements
def EuclideanDistances(A, B):
    assert len(A.shape) == 2 and len(B.shape) == 2
    assert A.shape[1] == B.shape[1]
    Asq = (np.linalg.norm(A, axis=1) ** 2)[:, None]
    BTsq = np.transpose((np.linalg.norm(B, axis=1) ** 2)[:, None])
    A_torch = torch.from_numpy(A)
    BT_torch = torch.from_numpy(B).T
    minux_2_AB = -2 * A_torch @ BT_torch
    plus = torch.from_numpy(Asq) + torch.from_numpy(BTsq)
    return (plus + minux_2_AB).pow(0.5)


def EuclideanDistances_torch(A, B):
    assert len(A.shape) == 2 and len(B.shape) == 2
    assert A.shape[1] == B.shape[1]
    assert A.dtype == np.float32
    A_torch = torch.from_numpy(A)
    B_torch = torch.from_numpy(B)
    Asq = torch.norm(A_torch, dim=1).pow(2)[:, None]
    BTsq = torch.norm(B_torch, dim=1).pow(2)[:, None]
    plus = Asq + BTsq.T
    # print(plus)
    minus_2AB = -2 * A_torch @ B_torch.T
    # print(plus + minus_2AB)
    # print((plus + minus_2AB).sqrt())
    res = (plus + minus_2AB).sqrt()
    # print(torch.from_numpy(np.zeros([2, 3])).pow(0.5))
    return res.numpy()
    # print(Asq)
    # print(BTsq)
    # print(plus)

    # Asq = (np.linalg.norm(A, axis=1) ** 2)[:, None]
    # BTsq = np.transpose((np.linalg.norm(B, axis=1) ** 2)[:, None])
    # A_torch = torch.from_numpy(A)
    # BT_torch = torch.from_numpy(B).T
    # minux_2_AB = -2 * A_torch @ BT_torch
    # plus = torch.from_numpy(Asq) + torch.from_numpy(BTsq)
    # return (plus + minux_2_AB).pow(0.5)


def EuclideanDistances_loop(A, B):
    assert len(A.shape) == 2 and len(B.shape) == 2
    assert A.shape[1] == B.shape[1]
    A_size = A.shape[0]
    B_size = B.shape[0]
    res = np.zeros([A_size, B_size])
    for i in range(A_size):
        for j in range(B_size):
            res[i, j] = np.linalg.norm(A[i] - B[j])

    return res


def EuclideanDistances_block(state):
    # state, action = get_paths(path_dir_path)
    items = state.shape[0]

    split_block = 10000
    split = int(items / split_block) + 1
    print(f"item {items}, split {split}")
    state_array = []
    final_res = np.zeros([items, items])
    for i in range(split):

        st = split_block * i
        ed = min(split_block * (i + 1), items)
        print(f"split {i}, from {st} to {ed}")
        cur_state = state[st:ed]
        state_array.append(cur_state)

    i_st = 0
    for i in range(split):
        j_st = 0
        i_size = state_array[i].shape[0]
        for j in range(split):
            j_size = state_array[j].shape[0]
            final_res[
                i_st : i_st + i_size, j_st : j_st + j_size
            ] = EuclideanDistances_torch(state_array[i], state_array[j])
            j_st += state_array[j].shape[0]
        print(f"split {i} segment calculated done")
        i_st += i_size
    final_res[np.isnan(final_res)] = 0
    return final_res


def find_top_nearest_k(K, state_array, distance_array):
    num_of_item = state_array.shape[0]
    assert distance_array.shape[0] == distance_array.shape[1] == num_of_item
    top_idx_array = []
    top_nearest_dist_array = []
    for i in range(num_of_item):
        # if i % 10 != 0:
        #     continue
        cur_state = state_array[i]
        cur_distance = distance_array[i]
        # print(cur_state)
        # torch_v, torch_idx = torch.topk(
        #     torch.from_numpy(cur_distance), K + 1, largest=False
        # )
        idx = np.argpartition(cur_distance, K + 1)[: K + 1]

        # print(f"old idx {idx}")
        idx = idx.tolist()
        if i in idx:
            idx.remove(i)

        # print(f"new idx {idx}")
        # idx = np.array().astype(np.int)
        dist = cur_distance[idx]
        top_idx_array.append(idx)
        top_nearest_dist_array.append(dist)
    return top_idx_array, top_nearest_dist_array

    # print(f"idx = {idx}")
    # print(f"dist = {dist}")
    # print(np.argpartition(cur_distance, -4))


# A = np.random.rand(40000, 50).astype(np.float32)
# EuclideanDistances_block(A, A)
# np.set_printoptions(linewidth=np.inf)
# np.set_printoptions(threshold=sys.maxsize)
max_traj = int(3e4)
path_state, path_action = get_paths(path_dir_path)
traj_state, traj_action = get_train_data(traj_summary_path)
# path_state = path_state[:200]
# traj_state = traj_state[:200]
# state = state[:200]
path_state = path_state[0:max_traj]
path_action = path_action[0:max_traj]
traj_state = traj_state[0:max_traj]
traj_action = traj_action[0:max_traj]
print(f"path state shape {path_state.shape}")
print(f"traj state shape {traj_state.shape}")
# exit()
path_distance = EuclideanDistances_block(path_state)
traj_distance = EuclideanDistances_block(traj_state)

# test = np.array([np.random.randint(10) for i in range(10)])
# print(test)
# print(np.argpartition(test, 4))
# print(np.argpartition(test, 4)[:4])
# print(test[np.argpartition(test, 4)[:4]])
K = 10
path_idx_array, path_dist_array = find_top_nearest_k(K, path_state, path_distance)
traj_idx_array, traj_dist_array = find_top_nearest_k(K, traj_state, traj_distance)

# 1. draw path
import matplotlib.pyplot as plt

# print(traj_action.shape[1])

if __name__ == "__main__":

    traj_action_std_mean = np.zeros(traj_action.shape[1])
    traj_action_mean_mean = np.zeros(traj_action.shape[1])
    for i in traj_idx_array:
        # print(i)
        cur_action = traj_action[i]
        # print(cur_action)
        traj_action_std_mean += np.std(cur_action, axis=0)
        traj_action_mean_mean += np.mean(cur_action, axis=0)

        # traj_traj_std_mean += std
    traj_action_std_mean /= len(traj_idx_array)
    traj_action_mean_mean /= len(traj_idx_array)
    plt.subplot(2, 2, 1)
    plt.plot(traj_action_std_mean, label="contact aware action std")
    plt.legend()
    plt.subplot(2, 2, 2)

    plt.plot(traj_action_mean_mean, label="contact aware action mean")
    plt.legend()
    print(f"traj action std {traj_action_std_mean}")
    print(f"traj action mean {traj_action_mean_mean}")

if __name__ == "__main__":

    path_action_std_mean = np.zeros(path_action.shape[1])
    path_action_mean_mean = np.zeros(path_action.shape[1])
    for i in path_idx_array:
        # print(i)
        cur_action = path_action[i]
        # print(cur_action)
        path_action_std_mean += np.std(cur_action, axis=0)
        path_action_mean_mean += np.mean(cur_action, axis=0)
        # path_path_std_mean += std
    path_action_std_mean /= len(path_idx_array)
    path_action_mean_mean /= len(path_idx_array)
    print(f"path action std {path_action_std_mean}")
    print(f"path action mean {path_action_mean_mean}")
    plt.subplot(2, 2, 3)
    plt.plot(path_action_std_mean, label="raw sampled action std")
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(path_action_mean_mean, label="raw sampled action mean")
    plt.legend()

plt.show()
# plt.subplot(1, 2, 1)
# 1.1 calc the std
# print(
#     f"traj std {np.sum([np.std(traj_action[i]) for i in traj_idx_array], axis=1) / len(traj_idx_array)}"
# )
# print(
#     f"path std {np.sum([np.std(path_action[i]) for i in path_idx_array], axis=1) / len(path_idx_array)}"
# )
# print(np.sum([np.std(traj_action[i]) for i in traj_idx_array], axis=1) / len(traj_idx_array)


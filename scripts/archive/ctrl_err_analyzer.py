import os
import numpy as np
import json
import matplotlib.pyplot as plt

skeleton_path = (
    "/home/xudong/Projects/DeepMimic/data/0805/characters/skeleton_080501.json"
)

log_path = "/home/xudong/Projects/bullet3/build_cmake/contact_aware.log"
DOF_MAP = {"none": 6, "revolute": 1, "spherical": 3}


def get_dof_offset(skeleton):
    with open(skeleton, "r") as f:
        root = json.load(f)
    joints = root["Skeleton"]["Joints"]
    name_lst = []
    dof_lst = []
    offset_lst = []
    cur_offset = 0
    for cur_joint in joints:
        name = cur_joint["Name"]
        joint_type = cur_joint["Type"]
        dof = DOF_MAP[joint_type]
        name_lst.append(name)
        dof_lst.append(dof)
        offset_lst.append(cur_offset)
        cur_offset += dof
    return name_lst, dof_lst, offset_lst, cur_offset


name_lst, dof_lst, offset_lst, total_dof = get_dof_offset(skeleton_path)


def get_difference(log_path, total_dof):
    q_diff_lst, qdot_diff_lst, qddot_diff_lst = [], [], []
    with open(log_path) as f:
        for line in f.readlines():
            if -1 != line.find("qddot diff"):
                qddot_diff_lst.append(np.array(line.strip().split()[4:]))
                assert qddot_diff_lst[-1].shape[0] == total_dof
            elif -1 != line.find("qdot diff"):
                qdot_diff_lst.append(np.array(line.strip().split()[4:]))
                assert qdot_diff_lst[-1].shape[0] == total_dof
            elif -1 != line.find("q diff"):
                q_diff_lst.append(np.array(line.strip().split()[4:]))
                assert q_diff_lst[-1].shape[0] == total_dof
    return q_diff_lst, qdot_diff_lst, qddot_diff_lst


q_diff_lst, qdot_diff_lst, qddot_diff_lst = get_difference(log_path, total_dof)


def show(diff_lst, joint_names, dofs, offsets, label, y_lim):
    # print(diff_lst)
    diff_mat = np.array(diff_lst)
    # print(np.shape(diff_mat))
    num_of_joints = len(joint_names)
    fig = plt.figure()
    for joint_id in range(num_of_joints):

        offset = offsets[joint_id]
        name = joint_names[joint_id]
        dof = dofs[joint_id]
        joint_col = diff_mat[:, offset: offset + dof]

        plt.subplot(3, 6, joint_id + 1)
        plt.ylim(y_lim)
        plt.plot(np.linalg.norm(joint_col, axis=1), label=f"{name}")
        plt.legend()
    # plt.tight_layout()
    fig.suptitle(f"{label} diff norm for each joint")


show(q_diff_lst, name_lst, dof_lst, offset_lst, "q", [0, 1])
show(qdot_diff_lst, name_lst, dof_lst, offset_lst, "qdot", [0, 1])
show(qddot_diff_lst, name_lst, dof_lst, offset_lst, "qddot", [0, 1])
plt.show()

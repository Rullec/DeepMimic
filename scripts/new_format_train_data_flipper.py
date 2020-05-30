import numpy as np
import json
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
from new_format_train_data_visualizer import action_minimum_analyze, action_axis_angle_theta_extract
st_num = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 46, 50, 55]


def flipper_single_traj_file(origin_filename, target_filename):
    assert(os.path.exists(origin_filename))
    
    with open(origin_filename, 'r') as f:
        root = json.load(f)
    

    # 1. extract the joint id data, then get the flipper interval 
    joint_axis_angle_theta_lst = action_axis_angle_theta_extract(origin_filename)
    # print(joint_axis_angle_theta_lst)
    joint_axis_angle_flipper_frame_lst = action_minimum_analyze(joint_axis_angle_theta_lst)
    

    for joint_id in range(len(st_num)):

        # 1. we need to flip the action for this joint (joint_id) in these frames (intervals)
        cur_joint_flipper_frame_lst = joint_axis_angle_flipper_frame_lst[joint_id]

        # print("joint %d flip from %d to %d" % (joint_id, st_num[joint_id]-1, st_num[joint_id]+3))
        # 2. for these intervals
        assert(len(cur_joint_flipper_frame_lst) % 2 == 0)
        for int_id in range(0, len(cur_joint_flipper_frame_lst), 2):
            frame_st = cur_joint_flipper_frame_lst[int_id]
            frame_ed = cur_joint_flipper_frame_lst[int_id+1]

            for cur_frame in range(int(frame_st), int(frame_ed)):
                cur_action = root["data_list"][cur_frame]["action"]
                # print("action %s" % str(cur_action))
                flip_pos_st = st_num[joint_id]-1
                # print("flip st pos %d " % flip_pos_st)
                raw_aa_data = cur_action[flip_pos_st:flip_pos_st+4]
                # print("flip raw data %s " % str(raw_aa_data))
                flip_aa_data = (-np.array(raw_aa_data)).tolist()
                # print("joint %d frame %d from %s" % (joint_id, cur_frame, str(raw_aa_data)))
                cur_action[flip_pos_st:flip_pos_st+4] = flip_aa_data
                # print("joint %d frame %d to %s" % (joint_id, cur_frame, str(flip_aa_data)))
                root["data_list"][cur_frame]["action"] = cur_action
                # exit(0)
    with open(target_filename, 'w') as f:
        json.dump(root, f)
        # print("write to %s" % target_filename)
    return

origin_train_data_dir = "/home/xudong/Projects/DeepMimic/data/batch_train_data/0526/"
target_train_data_dir = "/home/xudong/Projects/DeepMimic/data/batch_train_data/0526_flipped/"
summary_table_path = "/home/xudong/Projects/DeepMimic/data/batch_trajs/0526/summary_table_fullbody.json"

mpi_cnt = 0
def flipper_traj_file_mpi(file):
    global mpi_cnt
    target_filename = os.path.join(target_train_data_dir, file.split('/')[-1])
    print(" %d write to %s" % (mpi_cnt, target_filename.split('/')[-1]))
    mpi_cnt+=1
    flipper_single_traj_file(file, target_filename)

def migrate_the_summary_table(origin_summary_table_path, target_train_data_dir):
    assert(os.path.exists(origin_summary_table_path))
    assert(os.path.exists(target_train_data_dir))
    with open(origin_summary_table_path, 'r') as f:
        summary_root = json.load(f)
    
    trajs_lst = summary_root["single_trajs_lst"]
    for id in range(len(trajs_lst)):
        origin_train_data_file = trajs_lst[id]["train_data_filename"]
        target_train_data_file = os.path.join(target_train_data_dir, origin_train_data_file.split('/')[-1])
        trajs_lst[id]["train_data_filename"] = target_train_data_file
    summary_root["single_trajs_lst"] = trajs_lst

    target_summary_table_path = os.path.join(target_train_data_dir, origin_summary_table_path.split('/')[-1])
    with open(target_summary_table_path, 'w') as f:
        json.dump(summary_root, f, indent=4)
        print("write summary table to %s" % target_summary_table_path)


if __name__ == "__main__":

    migrate_the_summary_table(summary_table_path, target_train_data_dir)
    exit(0)
    # 1. load all datas and flipper them
    files = [str(os.path.join(origin_train_data_dir, i)) for i in os.listdir(origin_train_data_dir)]
    # files = files[ 0:10]
    # print(files)
    pool = Pool(12)
    pool.map(flipper_traj_file_mpi, files)
    # show_theta_for_new_train_data()

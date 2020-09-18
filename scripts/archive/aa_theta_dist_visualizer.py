import os
import numpy as np
import json
import matplotlib.pyplot as plt
from multiprocessing import Pool
from old_format_path_data_visualzer import st_num

mGuranty = 120 # resolution
paint_sample = 20
# fetch_axis = False
# fetch_theta = True
fetch_axis = True
fetch_theta = False

def get_new_data_action_theta_dist_with_ref_time_mpi(file_lst):
    global mGuranty
    assert(fetch_axis != fetch_theta)

    aa_theta_dist_with_respect_to_ref_time = np.zeros([len(st_num), mGuranty])
    # print(np.shape(aa_theta_dist_with_respect_to_ref_time))
    for single_file in file_lst:
        try:
            with open(single_file, 'r') as f:
                root = json.load(f)
        except:
            print("open %s failed" % single_file)
            continue
        # 1. load all states and actions
        states = [i["state"] for i in root["data_list"]][20:]
        actions = [i["action"] for i in root["data_list"]][20:]
        phases = [i[0] for i in states]
        num_of_frames = len(actions)

        # 2. begin to get the theta val for each joint in each frame
        for frame_id in range(num_of_frames):
            if frame_id % paint_sample == 0:
                cur_phase_int = int (phases[frame_id] * mGuranty)
                cur_action = actions[frame_id]

                if fetch_theta == True:
                    for sp_joint_id, aa_axis_st_pos in enumerate(st_num):
                        aa_theta_dist_with_respect_to_ref_time[sp_joint_id, cur_phase_int] += cur_action[aa_axis_st_pos-1]
                elif fetch_axis == True:
                    for sp_joint_id, aa_axis_st_pos in enumerate(st_num):
                        aa_theta_dist_with_respect_to_ref_time[sp_joint_id, cur_phase_int] += cur_action[aa_axis_st_pos]

                if frame_id % paint_sample == 0:
                    # print("[traj] frame %d joint 0 action theta = %.5f" % (frame_id//20, cur_action[0]))
                    continue
                    # if sp_joint_id == 2 and cur_phase_int < mGuranty / 10 and cur_action[aa_theta_st_pos] > 0.1:
                    #     print("traj %s phase %.5f joint %d action = %.5f"% ( single_file, phases[frame_id], sp_joint_id, cur_action[aa_theta_st_pos]))
                    #     print("full action = " + str(cur_action[0]))
                        # exit(0)


    return aa_theta_dist_with_respect_to_ref_time / len(file_lst)

def get_old_data_action_theta_dist_with_ref_time(file_lst):
    global mGuranty

    aa_theta_dist_with_respect_to_ref_time = np.zeros([len(st_num), mGuranty])
    # print(np.shape(aa_theta_dist_with_respect_to_ref_time))
    for single_file in file_lst:
        try:
            with open(single_file, 'r') as f:
                root = json.load(f)
        except:
            print("parse json %s failed, ignore" % single_file)
            continue
        
        # 1. load all states and actions
        states = root["states"][1:]
        actions = root["actions"][1:]
        phases = [i[0] for i in states]
        num_of_frames = len(actions)

        # 2. begin to get the theta val for each joint in each frame
        for frame_id in range(num_of_frames):
            cur_phase_int = int (phases[frame_id] * mGuranty)
            cur_action = actions[frame_id]


            if fetch_theta == True:
                for sp_joint_id, aa_axis_st_pos in enumerate(st_num):
                    aa_theta_dist_with_respect_to_ref_time[sp_joint_id, cur_phase_int] += cur_action[aa_axis_st_pos-1]
            elif fetch_axis == True:
                for sp_joint_id, aa_axis_st_pos in enumerate(st_num):
                    aa_theta_dist_with_respect_to_ref_time[sp_joint_id, cur_phase_int] += cur_action[aa_axis_st_pos]
                        

            # for sp_joint_id, aa_axis_st_pos in enumerate(st_num):
            #     aa_theta_dist_with_respect_to_ref_time[sp_joint_id, cur_phase_int] += cur_action[aa_axis_st_pos-1]
            #     if sp_joint_id == 0:
            #         # print("[path] frame %d joint 0 action theta = %.5f" % (frame_id, cur_action[aa_axis_st_pos-1]))
            #         continue

    return aa_theta_dist_with_respect_to_ref_time / len(file_lst)

        


old_format_data_dir = "../data/paths_normalized/"
new_format_data_dir = "../data/batch_train_data/0609_debug_solved/"
if __name__ == "__main__":
    old_format_files = [os.path.join(old_format_data_dir, i) for i in os.listdir(old_format_data_dir)]
    new_format_files = [[os.path.join(new_format_data_dir, i)] for i in os.listdir(new_format_data_dir) if i.find("train") != -1]
    # new_format_files = [os.path.join(new_format_data_dir, i) for i in os.listdir(new_format_data_dir) if i.find("train") != -1]


    # 1. find all old data, load their phase-action_theta map for each joint
    old_dist = get_old_data_action_theta_dist_with_ref_time(old_format_files)

    pool = Pool(12)
    new_dist_mpi_lst = pool.map(get_new_data_action_theta_dist_with_ref_time_mpi, new_format_files)
    new_dist = np.zeros_like(new_dist_mpi_lst[0])
    for cur_dist in new_dist_mpi_lst:
        new_dist += cur_dist / len(new_dist_mpi_lst)
    print("done")
    print(len(new_dist))
#     # old_dist: [sp_joints_num, Guranty]
    rows = 4
    cols = 4
    # rows = 1
    # cols = 1
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            plt_id = (row - 1) * rows + col

            joint_id = plt_id - 1
            if joint_id  == (np.shape(old_dist)[0]):
                break
        
            plt.subplot(rows, cols, plt_id)
            plt.plot(old_dist[joint_id], alpha = 0.5)
            plt.plot(new_dist[joint_id], alpha = 0.5)
            plt.legend(["ground_truth", "ID_res"])
            plt.title("joint %d action axis angle theta info" % joint_id, y=0.05)
    plt.show()
    

#     # 2. find all new data, load their phase-action_theta map for each joint
    
#     # 3. draw and paint them

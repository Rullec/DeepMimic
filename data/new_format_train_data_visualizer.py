import numpy as np
import json
import os
import matplotlib.pyplot as plt

# train_data_dir = "/home/xudong/Projects/DeepMimic/data/batch_train_data/0526/"
st_num = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 46, 50, 55]


def action_axis_angle_theta_extract(file):
    if -1 == file.find("train"):
        return
    joint_axis_angle_theta_lst = [[] for i in range(len(st_num))]
    with open(file, 'r') as f:
        data_list = json.load(f)["data_list"]
    for frame_id, single_frame in enumerate(data_list):
        single_action = single_frame["action"]
        if len(single_action) == 0:
            continue
            
        for id, joint_st in enumerate(st_num) :
            joint_axis_angle_theta_lst[id].append(single_action[joint_st-1])
        
    return joint_axis_angle_theta_lst

def draw_theta(joint_axis_angle_theta_lst, interval_lst = None):
    rows = 4
    cols = 4
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            plt_id = (row - 1) * rows + col

            joint_id = plt_id - 1
            if joint_id  == len(joint_axis_angle_theta_lst):
                break
        
            plt.subplot(rows, cols, plt_id)
            plt.plot(joint_axis_angle_theta_lst[joint_id])
            
            plt.title("joint %d action axis angle theta info" % st_num[joint_id], y=0.05)
            # plt.plot([i for i in range(1000)])

            if interval_lst is not None:
                interval_pt_lst = interval_lst[joint_id]
                if len(interval_pt_lst) is 0:
                    continue
                print("joint %dth %s" % (joint_id, str(interval_pt_lst)))
                
                for i in range(0, len(interval_pt_lst), 2): 
                    plt.plot(interval_pt_lst[i:i+2], [0, 0])
                    # print("draw %s" % str(interval_pt_lst[i:i+2]))
            

def list_smooth(num_lst, coef = 0.99):
    assert(type(num_lst) is list)
    assert(len(num_lst) == len(st_num))

    final_result_smooth_lst = []
    for single_joint_lst in num_lst:
        new_lst = []
        old_value = 0
        for i in range(len(single_joint_lst)):
            if 0 == i:
                old_value = single_joint_lst[0]
            else:
                old_value = single_joint_lst[i] * (1 - coef) + old_value * coef 
            new_lst.append(old_value)
        final_result_smooth_lst.append(new_lst)
    return final_result_smooth_lst

def action_minimum_analyze(smooth_data_lst):
    threshold = 7e-2
    # 1. get the continuous data interval that all datas are smaller than threshold
    full_interval_lst = []
    for joint_id, single_joint_lst in enumerate(smooth_data_lst):
        InTheInterval = False
        interval_st = -1
        interval_ed = -1
        interval_mid_lst = []
        for id, joint_theta_val in enumerate(single_joint_lst):
            # assert(joint_theta_val > 0)
            if InTheInterval == False:
                if joint_theta_val < threshold:
                    interval_st = id
                    InTheInterval = True
                else:
                    continue
            elif InTheInterval == True:
                if joint_theta_val < threshold:
                    continue
                else:
                    interval_ed = id - 1
                    interval_mid_lst.append((interval_ed + interval_st) / 2)
                    # print("joint %d mid pt %d" % (st_num[joint_id], interval_mid_lst[-1]))
                    InTheInterval = False
        if InTheInterval == True:
            interval_ed = len(single_joint_lst) - 1
            interval_mid_lst.append((interval_st + interval_ed) / 2)

        # 2. convert these pts into intervals
        interl_endpt_lst = []
        if len(interval_mid_lst) % 2 == 1:
            interval_mid_lst.insert(0, 0)
        for i in range(0, len(interval_mid_lst), 2):
            # print("idx %d, len %d" % (i, len(interval_mid_lst)))
            interl_endpt_lst += [interval_mid_lst[i], interval_mid_lst[i+1]]

        full_interval_lst.append(interl_endpt_lst)
    return full_interval_lst

# def flip_data_by_interval_lst(joint_data, int_lst):
def flip_theta_lst(joints_theta_lst, joints_flip_interval_lst):
    joint_num = len(st_num)
    assert(len(joints_flip_interval_lst) == joint_num)
    assert(len(joints_theta_lst) == joint_num)

    flipped_joints_theta_lst = []
    for joint_id in range(len(st_num)):
        single_joint_flip_int = joints_flip_interval_lst[joint_id]
        single_joint_theta_lst = joints_theta_lst[joint_id]
        
        assert(len(single_joint_flip_int) %2 == 0)
        if len(single_joint_flip_int) != 0:
            for i in range(0, len(single_joint_flip_int), 2):
                frame_st = single_joint_flip_int[i]
                frame_ed = single_joint_flip_int[i+1]
                print("joint %d flipped from %d to %d" % (joint_id, frame_st, frame_ed))
                for j in range(int(frame_st), int(frame_ed)+1):
                    single_joint_theta_lst[j] = -1 * single_joint_theta_lst[j]
    
        flipped_joints_theta_lst.append(single_joint_theta_lst)
    return flipped_joints_theta_lst

train_data_dir = "/home/xudong/Projects/DeepMimic/data/batch_train_data/0526/"
train_data_flipped_dir = "/home/xudong/Projects/DeepMimic/data/batch_train_data/0526_flipped/"

def show_theta_for_new_train_data():
    # for i in range(10):
    # show raw result
    # files = [os.path.join(train_data_dir, i) for i in os.listdir(train_data_dir)]
    # joint_axis_angle_theta_lst = action_axis_angle_theta_extract(files[0])
    # joint_minimum_interval_pt_lst = action_minimum_analyze(joint_axis_angle_theta_lst)
    # draw_theta(joint_axis_angle_theta_lst, joint_minimum_interval_pt_lst)
    # plt.show(block=False)

    # show now 
    files = [os.path.join(train_data_flipped_dir, i) for i in os.listdir(train_data_flipped_dir)]
    joint_axis_angle_theta_lst = action_axis_angle_theta_extract(files[0])
    draw_theta(joint_axis_angle_theta_lst, None)

    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()
    return

if __name__ == "__main__":
    show_theta_for_new_train_data()
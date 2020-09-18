import numpy as np
import json
import os
import matplotlib.pyplot as plt

'''
joint 1 Spine1 aa st from 1
joint 2 vneck aa st from 5
joint 3 RightArm aa st from 9
joint 4 RightForeArm aa st from 13
joint 5 RightHand aa st from 17
joint 6 Neck aa st from 21
joint 7 Head aa st from 25
joint 8 LeftArm aa st from 29
joint 9 LeftForeArm aa st from 33
joint 10 LeftHand aa st from 37
joint 11 RightLeg aa st from 41
joint 13 RFootTongue aa st from 46
joint 14 LeftLeg aa st from 50
joint 16 LFootTongue aa st from 55
'''
st_num = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 46, 50, 55]
# origin_path_dir = "./paths"
origin_path_dir = "../data/paths"
# origin_path_dir = "./paths_normalized"
# origin_path_dir = "./paths_ori_single"
files = [os.path.join(origin_path_dir, i) for i in os.listdir(origin_path_dir)]

def calc_axis_angle_theta_lst(action):
    global st_num
    assert(type(action) is list)
    action = np.array(action)
    axis_angle_theta_lst = []
    for id, cur_st in enumerate(st_num):
        res = np.linalg.norm(action[cur_st-1])
        assert(res > 0)
        axis_angle_theta_lst.append(res)
    return axis_angle_theta_lst


def calc_axis_angle_norm_lst(action):
    global st_num
    assert(type(action) is list)
    action = np.array(action)
    axis_angle_norm_lst = []
    for id, cur_st in enumerate(st_num):
        res = np.linalg.norm(action[cur_st : cur_st + 3])
        assert(res > 0)
        axis_angle_norm_lst.append(res)
    return axis_angle_norm_lst

def action_axis_angle_norm_extract(files,max_file = 10):
    '''
        Here are lots of axis angle representation in a single "action"
        this function will calculate 
        return: 
            [
                axis_angle1_norm : [frame1_val, frame2_val, ..., frameN_val]
                ...
                axis_angleN_norm : [frame1_val, frame2_val, ..., frameN_val]
            ]

        "aa" is abbreavited of "axis angle"
    '''
    global st_num
    return_aa_norm_lst_total_trajs = []
    return_aa_theta_lst_total_trajs = []
    for file_id, file in enumerate(files):
        if file_id == max_file:
            break

        if -1 == file.find("json"):
            continue
        with open(file, 'r') as f:
            action_lst = json.load(f)["actions"]
            return_aa_norm_lst_single_traj = [ [] for i in range(len(st_num))]
            return_aa_theta_lst_single_traj = [ [] for i in range(len(st_num))]
            for single_action in action_lst:
                single_action_aa_norm = calc_axis_angle_norm_lst(single_action)
                single_action_aa_theta = calc_axis_angle_theta_lst(single_action)
                # print(single_action_aa_norm)
                assert(len(single_action_aa_norm) == len(st_num))
                for j in range(len(st_num)):
                    return_aa_norm_lst_single_traj[j].append(single_action_aa_norm[j])
                    return_aa_theta_lst_single_traj[j].append(single_action_aa_theta[j])
            return_aa_norm_lst_total_trajs.append(return_aa_norm_lst_single_traj)
            return_aa_theta_lst_total_trajs.append(return_aa_theta_lst_single_traj)
    return return_aa_norm_lst_total_trajs, return_aa_theta_lst_total_trajs

def show_axis_norm():
    '''
        paint joint axis norm
    '''
    axis_angle_norm_lst, axis_angle_theta_lst = action_axis_angle_norm_extract(files, max_file=8)
    stop = 100
    for i in range(len(axis_angle_norm_lst)):
        joint_aa_norm_avg_lst = []
        plt.subplot(1, 2, 2)
        for j, cur_aa_norm_lst in enumerate(axis_angle_norm_lst[i]):
            if j == stop:
                break
            plt.plot(cur_aa_norm_lst)
            avg = np.average(cur_aa_norm_lst)
            var = np.var(cur_aa_norm_lst)
            # print(cur_aa_norm_lst)
            print("for joint %d, axis norm avg = %.3f, var = %.3f" % (j, avg, var))
        print("for all joints, axis norm avg = %.3f, var = %.3f" % ( np.average(axis_angle_norm_lst[i]), np.var(axis_angle_norm_lst[i])))
    
        plt.legend([str(i) for i in st_num])
        plt.xlabel("frame id")
        plt.ylabel("joint axis angle norm")
        plt.title("the norm of axis angles in DeepMimic walk controller's action")

        plt.subplot(1, 2, 1)
        for j, cur_aa_thet_lst in enumerate(axis_angle_theta_lst[i]):
            if j == stop:
                break
            plt.plot(cur_aa_thet_lst)
            avg = np.average(cur_aa_thet_lst)
            var = np.var(cur_aa_thet_lst)
            # print(cur_aa_norm_lst)
            print("for joint %d, axis theta avg = %.3f, var = %.3f" % (j, avg, var))
        print("for all joints, axis theta avg = %.3f, var = %.3f" % ( np.average(axis_angle_theta_lst[i]), np.var(axis_angle_theta_lst[i])))
    
        plt.legend([str(i) for i in st_num])
        plt.xlabel("frame id")
        plt.ylabel("joint axis angle theta")
        plt.title("the theta of axis angles in DeepMimic walk controller's action")

        plt.show()
        exit(0)

def action_axis_angle_theta_extract(file_lst):
    for file_id, file in enumerate(files):
        if -1 == file.find("json"):
            continue
        joint_axis_angle_theta_lst = [[] for i in range(len(st_num))]
        with open(file, 'r') as f:
            action_lst = json.load(f)["actions"]
        for single_action in action_lst:
            for id, joint_st in enumerate(st_num):
                joint_axis_angle_theta_lst[id].append(single_action[joint_st-1])
        return joint_axis_angle_theta_lst

def action_axis_angle_digit_extract(file_lst):
    global st_num
    for file_id, file in enumerate(files):
        if -1 == file.find("json"):
            continue
        joint_axis_digits = [[] for i in range(len(st_num))]
        with open(file, 'r') as f:
            action_lst = json.load(f)["actions"]
        for single_action in action_lst:
            for id, joint_st in enumerate(st_num):
                joint_axis_digits[id].append(single_action[joint_st])
                joint_axis_digits[id].append(single_action[joint_st + 1])
                joint_axis_digits[id].append(single_action[joint_st + 2])
        return joint_axis_digits

def show_axis_per_digit():
    joint_axis_digits = action_axis_angle_digit_extract(files)
    print(len(joint_axis_digits))
    rows = 4
    cols = 4
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            plt_id = (row - 1) * rows + col

            joint_id = plt_id - 1
            if joint_id  == len(joint_axis_digits):
                break
            
            plt.subplot(rows, cols, plt_id)
            cur_joint_seq = joint_axis_digits[joint_id]
            # print(cur_joint_seq)
            plt.plot(cur_joint_seq[0::3])
            plt.plot(cur_joint_seq[1::3])
            plt.plot(cur_joint_seq[2::3])
            # plt.plot([ np.linalg.norm(cur_joint_seq[i:i+3]) for i in range(0, len(cur_joint_seq), 3)])
            # plt.legend(["x", "y", "z", "norm"])
            plt.legend(["x", "y", "z"])
            plt.title("joint %d action axis 3 digits info" % st_num[joint_id], y=1)
            # plt.xlabel("frame")
            plt.ylabel("val")
            # plt.plot([i for i in range(10)])
    plt.title("show buffer in %s" % origin_path_dir)
    plt.show()
    # for id in range(len(st_num)):
        
def show_theta():
    joint_axis_angle_theta_lst = action_axis_angle_theta_extract(files)
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
    plt.show()
    return


if __name__ == "__main__":
    # show_axis_per_digit()
    # show_axis_norm()
    show_theta()

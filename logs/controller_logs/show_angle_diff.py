import os, sys
import re
import numpy as np
import matplotlib.pyplot as plt

name = '''root 0
rhipjoint 1
rfemur 2
rtibia 3
rfoot 4
lowerback 5
upperback 6
thorax 7
rclavicle 8
rhumerus 9
rradius 10
rwrist 11
lowerneck 12
lclavicle 13
lhumerus 14
lradius 15
lwrist 16
lhipjoint 17
lfemur 18
ltibia 19
lfoot 20
'''
name = name.split()
id_name_map = {}
for i in range(0, len(name), 2):
    id_name_map[int(i/2)] = name[i]


def read_one_joint(joint_id, angle_diff_dir):
    file_name = os.path.join(angle_diff_dir, "%d.txt" % joint_id)
    try:
        with open(file_name, "r") as f:
            cont = f.read().splitlines()
    except:
        print("read %s failed, jumped" % file_name)

    def find_floats(string):
        raw_lst = re.findall(r"[+-]?([0-9]*\.?[0-9]+|[0-9]+\.?[0-9]*)([eE][+-]?[0-9]+)?", string)
        lst = []
        for a, b in raw_lst:
            if len(b) == 0:
                lst.append(float(a))
            else:
                a = float(a) * pow(10, float(b[1:]))
                lst.append(a)
        # print(lst)
        return lst

    # handle these conts
    angle_diff = []
    vel_diff = []
    for line in cont:
        if line.find("pose") != -1:
            time_str, id_str, cur_str, motion_str = line.split(",")
            angle_diff.append(np.linalg.norm(np.array(find_floats(cur_str)) - find_floats(motion_str)))

        elif line.find("vel") != -1:
            time_str, id_str, cur_str, motion_str = line.split(",")
            vel_diff.append(np.linalg.norm(np.array(find_floats(cur_str)) - find_floats(motion_str)))

        else:
            raise (ValueError)
    # print(cont)
    return angle_diff, vel_diff

def paint_dir(dir_path):
    dir_path = os.path.abspath(dir_path)
    files = os.listdir(dir_path)
    num = len(files)
    print(num)
    plt.rcParams["figure.figsize"] = (20, 10)

    fig = plt.figure(num)
    for i in range(num):
        angle, vel = read_one_joint(i, dir_path)
        plt.subplot(5, 5, i + 1)
        plt.plot(angle)
        plt.plot(vel)
        plt.title(id_name_map[i])
        plt.legend(["angle", "vel"])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    args = sys.argv[1:]
    for i in args:
        paint_dir(i)

import os
import numpy as np
import sys
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

if __name__ == "__main__":
    files = sys.argv[1:]
    assert len(files) == 1
    file = files[0]
    print("begin to analysis {}".format(file))

    cont = []
    with open(file,"r") as f:
        cont = f.readlines()
        cont = [i.strip() for i in cont]
    num = len(id_name_map)
    data_lst = [[] for i in range(num)]
    for i in cont:
        _, id, _, _, a, b, c, d = i.split()
        data_lst[int(id[:-1])].append(np.linalg.norm(np.array([float(a), float(b), float(c), float(d)])))
    # print(data_lst)

    plt.rcParams["figure.figsize"] = (20, 10)

    fig = plt.figure(num)
    for i in range(num):
        plt.subplot(5, 5, i + 1)
        plt.plot(data_lst[i])
        plt.title(id_name_map[i])
    plt.tight_layout()
    plt.show()
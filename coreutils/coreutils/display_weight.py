import sys
import json
import os
import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from coreutils.utils import read_batch_info

'''
    本文件是用来可视化不同的weight的
    对于相近的conf，他们的weight是否也应该相近? 需要用矩阵的形式表达出来
    1. 找到所有的conf
    2. 找到这些conf对应的weight
    3. 选定其中一个weight，看其他conf和他的关系
'''

def measure_2_weight(weight1, weight2):
    # name_lst = ["actor_net/sigma/weights:0",\
    #      "actor_net/sigma/biases:0",
    #      "actor_net/mu/weights:0",  
    #      "actor_net/mu/biases:0", ]
    dist = 0
    for name in weight1:
        dist += np.linalg.norm(weight1[name] - weight2[name])
        print("name:%s, shape:%s" % (name, weight1[name].shape))
    # print(dist)
    return dist

def measure_1_weight(weight):
    dist = 0
    for name in weight:
        dist += np.linalg.norm(weight[name])
    return dist

if __name__ == "__main__":
    # 获取model路径和conf路径
    model_path_lst, conf_path_lst = read_batch_info()

    # # 读model的weight
    model_weight_lst = []
    for path in model_path_lst:
        path += ".weight"
        assert os.path.exists(path) == True
        with open(path, "rb") as f:
            model_weight_lst.append(pickle.load(f))
    
    # 读model的conf
    conf_dict_lst = []
    m_lst = []
    l_lst = []
    for path in conf_path_lst:
        with open(path, "rb") as f:
            cont = json.load(f)
            m_lst.append(cont["m"])
            l_lst.append(cont["l"])

    # 绘制差距图
    target_id = 15
    dist_lst_1 = []
    dist_lst_2 = []
    for i in range(len(model_weight_lst)):
        dist2 = measure_2_weight(model_weight_lst[i], model_weight_lst[target_id])
        dist_lst_2.append(dist2)
        dist1 = measure_1_weight(model_weight_lst[i])
        dist_lst_1.append(dist1)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.plot_surface(m_lst, l_lst, dist_lst)
    ax.scatter3D(m_lst, l_lst, dist_lst_1, color="b")
    ax.scatter3D(m_lst[target_id], l_lst[target_id],\
         dist_lst_1[target_id], color="r")
    plt.show()

    # 绘制的单个图

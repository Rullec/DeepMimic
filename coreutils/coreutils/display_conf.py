import os
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import json
from coreutils.utils import read_batch_info

'''
    本文件是用来可视化不同的configuration的
'''

def write_single_conf(conf_dir_path, conf_id):
    '''
        为每个model dir写入conf文件
        conf_dir_path: 往哪个地方写入conf?
        conf_id: 这个conf的编号是什么?
    '''
    # 这个函数负责往conf_dir_path文件夹下面写conf
    # 这个逻辑随时需要重写，等等
    import re
    from operator import itemgetter
    para_list = ["conf_id", "m", "l"]
    
    # compute conf content
    conf_content = os.path.split(conf_dir_path)[-1]
    demand = [1, 3]
    m, l = itemgetter(*demand)(re.split(",|=", conf_content))
    m, l = float(m), float(l)
    
    conf_dict = {}
    for i in range(len(para_list)):
        conf_dict[para_list[i]] = locals()[para_list[i]]

    # write down the conf content to json
    # 命名规则后续一定需要重新写:
    conf_path = os.path.join(conf_dir_path, "m=%.2f,l=%.2f.conf" % (m, l))
    print("writing to %s: %s" % (conf_path, str(conf_dict)))
    with open(conf_path, "w") as f:
        json.dump(conf_dict, f)

def write_batch_conf(project_dir_path="/home/xudong/Projects/controller-space/data", \
        env_name_dir_lst = ["saved_ppo_models"], model_save_dir_lst = ["NewPendulum-v0"]):
    # 这个函数负责写conf，批量写conf
    id = 0
    for env_name_dir in env_name_dir_lst:
        for model_save_dir in model_save_dir_lst:
            sum_path = os.path.join(project_dir_path, env_name_dir, model_save_dir)
            if os.path.exists(sum_path) == False:
                raise(FileNotFoundError)
            for conf_dir in os.listdir(sum_path):
                conf_path = os.path.join(sum_path, conf_dir)
                write_single_conf(conf_path, id)
                id += 1

def select_key_pts(i, conf_lst,  k = 5):
    '''
        给出所有configuraion info的列表
        返回第i个的k=5近邻的id
    '''
    conf_mat = np.transpose(np.array(conf_lst))   # 现在conf_mat是一个70 * 18的矩阵
    
    target_vec = np.reshape(np.tile(conf_mat[i], conf_mat.shape[0]), conf_mat.shape)# 将conf_mat[i]，也就是第i个conf重复70次，也形成一个70*18的矩阵
    # print(conf_mat)
    # print(target_vec)
    norm2_res = np.linalg.norm(conf_mat - target_vec, axis=1)   # 两个矩阵相减，并且对axis=1求l2-norm，得到一个70*1个列表，这就是各个conf到第i个conf的距离。
    norm2_res_id = np.vstack([np.arange(0, len(conf_lst), 1), norm2_res])  # 给上面那个70*1的列表，附加上一个0-69的id信息，形成一个矩阵为2*70, 其中第一行为0-69的顺序id, 第二行为对应的距离
    # print(norm2_res_id)
    order = np.lexsort(norm2_res_id)    # 以第二行的数字为关键值，对这个2*70数组的70个列进行排序
    sorted_norm2 = norm2_res_id[:,order]
    
    # 获取距离最近的前k个id:
    id_res = sorted_norm2[0, 1:k+1]
    dist_res = sorted_norm2[1, 1:k+1]
    return id_res.astype(int).tolist(), dist_res.tolist()

if __name__ == "__main__":
    import time
    from operator import itemgetter
    write_batch_conf()
    # _, conf_path_lst = read_batch_info()
    # cont_dict_lst = []

    # for path in conf_path_lst:
    #     with open(path, "r") as f:
    #         cont_dict_lst.append(json.load(f))
            
    # m_lst = []
    # l_lst = []
    # for i in cont_dict_lst:
    #     # print(i)
    #     m_lst.append(i["m"])
    #     l_lst.append(i["l"])
    # # a = time.time()
    
    # plt.ion()
    # for i in range(len(m_lst)):
    #     pt_id_lst, _ = select_key_pts(i, m_lst, l_lst)
    #     # print(time.time() - a)
    #     plt.scatter(m_lst, l_lst)
    #     plt.scatter(itemgetter(*pt_id_lst)(m_lst), itemgetter(*pt_id_lst)(l_lst))
    #     plt.xlabel("mass")
    #     plt.ylabel("length")
    #     plt.pause(0.5)
    #     plt.cla()
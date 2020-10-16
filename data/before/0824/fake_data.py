import numpy as np
import json
import os
import pickle
# 这个文件是用来制造假数据的
'''
    1. 读取configuration 目录下面的所有文件
    2. 读取当前文件夹下面的weight，模仿格式为所有json都建立
    3. json输出为info.json文件，然后得到结果。
'''

conf_dir = os.path.join(os.getcwd(), "data0824/characters/")
models_dir = os.path.join(os.getcwd(), "data0824/models/")


def get_all_conf(conf_dir):
    def verify(name):
        assert type(name) == str
        return -1 != name.find("skeleton") and -1 != name.find("json")

    name_lst = os.listdir(conf_dir)
    name_lst.sort()
    ret = []
    for i in name_lst:
        if verify(i) == True:
            ret.append(i)
    return ret


if __name__ == "__main__":
    confs = get_all_conf(conf_dir)
    # 读取真正weight
    true_weight = os.path.join(models_dir, "example.ckpt.weight")
    with open(true_weight, "rb") as f:
        true_weight = pickle.load(f)
    names, shapes = [], []
    for i in true_weight.keys():
        names.append(i)
        shapes.append(true_weight[i].shape)

    def shape_a_new_weight(weight):
        new_weight = weight.copy()
        for i in new_weight.keys():
            new_weight[i] *= np.random.rand()
        return new_weight

    info_json = []
    for sub_conf in confs:
        new_weight = shape_a_new_weight(true_weight)
        new_weight_name = sub_conf.replace("json", "weight")

        item = {}
        item["skeleton"] = os.path.join(conf_dir, sub_conf)
        item["weight"] = os.path.join(models_dir, new_weight_name)
        info_json.append(item)

        with open(item["weight"], "wb") as f:
            pickle.dump(new_weight, f)
    info_json_path = os.path.join(models_dir, "info.json")
    with open(info_json_path, "w+") as f:
        json.dump(info_json, f, indent=4)

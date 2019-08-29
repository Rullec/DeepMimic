import os
import json
import csv
import sys
from verify_skeleton import verify

project_dir = "/home/darknight/Projects/DeepMimic"
skeleton_path = "data/0828/characters/skeleton_082801_1.json"
pd_path = "data/0828/controllers/humanoid3d_ctrl_082901.txt"

reduce = lambda f : round(f, 4)
def parse_pd(file):
    f_pd = open(file, "r")
    value = json.load(f_pd)
    info = {}
    for item in value["PDControllers"]:
        name = item["Name"]
        info[name] = {"Kp": item["Kp"], "Kd": item["Kd"]}
    return info

def parse_skeleton(file):
    '''
        this function will parse the skeleton json value,
        extract the info for each joint.

    '''
    f_skeleton = open(file, "r")
    value = json.load(f_skeleton)

    bodydefs_value = value["BodyDefs"]
    info_dict = {}
    key_lst = ["Name", "ID", "Mass", "ColGroup"]
    # Name, ID, Mass, Param0, Param1, Param2, ColGroup = (None,) * len(key_lst)
    for i in bodydefs_value:
        info = {}
        Name = i["Name"]
        for key in key_lst:
            info[key] = i[key]

        Param0, Param1, Param2 = i["Param0"],i["Param1"],i["Param2"]
        Volume = Param0 * Param1 * Param2
        Length = Param1
        info["Volume"] = Volume * 1e6
        info["Length"] = Length

        # add
        info_dict[Name] = info

    skeleton_joints = value["Skeleton"]["Joints"]
    for i in skeleton_joints:
        name = i["Name"]
        torquelim = reduce(i["TorqueLim"])
        diff_weight = reduce(i["DiffWeight"])
        type_ = i["Type"]

        # add
        info_dict[name]["TorqueLim"] = torquelim
        info_dict[name]["DiffWeight"] = diff_weight
        info_dict[name]["Type"] = type_
    for dict_key in info_dict.keys():
        item = info_dict[dict_key]
        for key in item.keys():
            if type(item[key]) == float:
                item[key] = reduce(item[key])
    return info_dict

def write_csv(info_dict, path = "data.csv"):
    # prepare
    if os.path.exists(path):
        os.remove(path)
    
    def write_tabular_line(lst, f):
        with open(path, "a") as f:
            for i, cont in enumerate(lst):
                f.write("%s" % cont)
                if i!= len(lst) - 1:
                    f.write(", ")
            f.write("\n")
    title_lst = list(info_dict["root"].keys())
    
    with open(path, "a") as f:
        write_tabular_line(title_lst, f)
        write_tabular_line(title_lst, f)

    # write content
    for cur_id in range(len(info_dict)):
        selected = None
        for i in info_dict.keys():
            if info_dict[i]["ID"] != cur_id:
                continue
            else:
                selected = info_dict[i]
                break

        # 对于选中的item: selcted, 写入信息
        cont = [selected[i] for i in title_lst]
        with open(path, "a") as f:
            write_tabular_line(cont, f)
    print("write csv to %s" % path)
        
if __name__ == "__main__":

    skeleton_path = os.path.join(project_dir, skeleton_path)
    pd_path = os.path.join(project_dir, pd_path)
    print("skeleton path = %s" % skeleton_path)
    print("pd path = %s" % pd_path)
    
    # 验证有效性
    verify(skeleton_path)

    # 解析skeleton，获取要写入的信息
    ske_info = parse_skeleton(skeleton_path)

    # 解析pd文件
    pd_info = parse_pd(pd_path)

    # 合并信息
    for item in ske_info:
        ske_info[item]["Kp"] = pd_info[item]["Kp"]
        ske_info[item]["Kd"] = pd_info[item]["Kd"]

    # 写入信息
    write_csv(ske_info)

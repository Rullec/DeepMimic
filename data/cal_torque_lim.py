import json
import os
import numpy as np
project_dir = "/home/xudong/Projects/DeepMimic"
skeleton_path = "data/0906/characters/skeleton_0907_torque_revised_cal.json"
# 本程序试图从skeleton的结构中推导出合适的torque lim
# 假定一个父joint的max torque lim = 其所有儿子link直立时产生的torque

if __name__ == "__main__":
    filename = os.path.join(project_dir, skeleton_path)
    f_skeleton = open(filename, "r")
    value = json.load(f_skeleton)
    joints_json = value["Skeleton"]["Joints"]   # skeleton json
    bodydefs_json  = value["BodyDefs"]  # body defs json
    
    # 填充好这５个lst以后才能计算
    num_joints  = len(joints_json)
    id_name_lst, id_mass_lst, id_length_lst, id_children_lst, id_parent_lst = \
        [None] * num_joints, [0.0] * num_joints, [0.0] * num_joints, [[] for i in range(num_joints)], [-1] * num_joints
    
    # read id, name, parent, length in Skeleton-Joints
    for joint in joints_json:
        id, name, parent = joint["ID"], joint["Name"], joint["Parent"]

        id_name_lst[id] = name
        if parent is not -1:
            # whose child is id ?
            # print("for id {}, parent {}".format(id, parent))
            id_children_lst[parent].append(id)
            # so, his parent's length = id[AttachX, AttachY, AttachZ, ].norm()
        
        id_length_lst[parent] = max(np.linalg.norm(np.array([joint["AttachX"], joint["AttachY"], joint["AttachZ"]])), id_length_lst[parent])
        id_parent_lst[id] = parent  # who is id's parent ?
    
    # read mass in BodyDefs
    for body in bodydefs_json:
        id, mass, name = body["ID"], body["Mass"], body["Name"]
        assert name == id_name_lst[id]
        id_mass_lst[id] = mass
    
    # build torque lim 
    id_torquelim = [0.0] * num_joints
    for i in range(num_joints):
        if(len(id_children_lst[i]) != 0):
            continue

        # if it is end effector
        parent = id_parent_lst[i]
        length_seq = [0.2]# end effector长度
        mass_seq = [id_mass_lst[i]]
        id_seq = [i]
        while parent != -1:
            # print(id_length_lst[parent])
            length_seq.append(id_length_lst[parent])
            mass_seq.append(id_mass_lst[parent])
            id_seq.append(parent)
            parent = id_parent_lst[parent]
        
        # 分别计算seq中各个子joint对他的torque lim 贡献
        for num in range(len(length_seq)):
            id = id_seq[num]
            length = sum(length_seq[:num]) + length_seq[num] / 2
            id_torquelim[id] += mass_seq[num] * length
    
    for i in range(num_joints):
        print("joint {}, name {}, mass {}, length {}, child {}, parent {}, torquelim {}".format(
            i, id_name_lst[i], id_mass_lst[i], id_length_lst[i], id_children_lst[i], id_parent_lst[i], id_torquelim[i]
        ))
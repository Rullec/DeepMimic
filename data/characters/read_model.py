import os
import json
import numpy as np
import queue
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ori_file = 'data/characters/humanoid3d_changed.txt' #
goal_file = ori_file + ".revised"

class Joint:
    # const state_key = ["ID", "Name", ""]

    def __init__(self):
        return

    def load_params(self, joint_dict):
        '''
            load joint info from a dict
        '''
        assert type(joint_dict) == dict

        self.__id = joint_dict["ID"]
        self.__name = joint_dict["Name"]
        self.__joint_type = joint_dict["Type"]
        self.__parent_id = joint_dict["Parent"]
        self.__local_pos = [joint_dict["AttachX"], joint_dict["AttachY"], joint_dict["AttachZ"]]
        self.__local_orient = [joint_dict["AttachThetaX"], joint_dict["AttachThetaY"], joint_dict["AttachThetaZ"]]
        self.__weight = joint_dict["DiffWeight"]
        self.__child_id = []
        print("""loat joint succ, id = %d, name = %s, joint_type = %s, parentid= %d, local_pos = %s, local_orient = %s, weight = %.3f"""
            %
            (self.__id, self.__name, self.__joint_type, self.__parent_id, self.__local_pos, self.__local_orient, self.__weight))

    def get_id(self):
        assert self.__id is not None
        return self.__id
    
    def get_name(self):
        assert self.__name is not None
        return self.__name

    def get_parent_id(self):
        assert self.__parent_id is not None
        return self.__parent_id
    
    def add_child_id(self, child_id):
        assert type(child_id) is int
        succ = False
        if False == (child_id in self.__child_id):
            self.__child_id.append(child_id)
            succ = True
        return succ
    
    def get_child_id(self):
        assert self.__child_id is not None
        return self.__child_id

    def set_global_pos(self, x, y, z):
        self.__global_pos = [x, y, z]

    def get_global_pos(self):
        if self.__global_pos is None:
            raise ValueError("no global pos in joint")
        else:
            return self.__global_pos

    def get_local_pos(self):
        if self.__local_pos is None:
            raise ValueError("no local pos in it")
        else:
            return self.__local_pos

class Model:
    def __init__(self):

        return

    def load_model(self, joint_json_list):
        assert type(joint_json_list) == list and len(joint_json_list) > 0
        self.__joint_list = [None] * len(joint_json_list)
        
        # arrange the model
        for i in joint_json_list:
            joint = Joint()
            joint.load_params(i)
            cur_id = joint.get_id()
            if self.__joint_list[cur_id] is not None:
                raise ValueError("the id of joints have conflicts!")
            self.__joint_list[cur_id] = joint
        
        # set child joint for each one
        for joint in self.__joint_list:
            parent_id = joint.get_parent_id()
            if parent_id == -1:
                self.__root_id = joint.get_id()
                continue
            if parent_id > len(self.__joint_list) or parent_id == joint.get_id():
                raise ValueError("joint id exceeds the bound")

            self.__joint_list[parent_id].add_child_id(joint.get_id())
        if self.__root_id is None:
            raise ValueError("there is no root joint!") 

        # compute global position
        q = queue.Queue()
        root_joint = self.__joint_list[self.__root_id]
        root_joint_global_pos = root_joint.get_local_pos()
        root_joint.set_global_pos(root_joint_global_pos[0], root_joint_global_pos[1], root_joint_global_pos[2])
        q.put(self.__root_id)

        while q.qsize() != 0:
            joint_id = q.get()
            joint = self.__joint_list[joint_id]
            joint_global_pos = joint.get_global_pos()
            childs = joint.get_child_id()
            for joint_id_child in childs:
                joint_child = self.__joint_list[joint_id_child]
                joint_local_pos_child = joint_child.get_local_pos()
                joint_global_pos_child = [joint_global_pos[i]+joint_local_pos_child[i] for i in range(3)]
                assert len(joint_global_pos_child) == 3
                joint_child.set_global_pos(joint_global_pos_child[0], joint_global_pos_child[1], joint_global_pos_child[2])
                q.put(joint_id_child)
            
        print("load model succ, joint num = %d" % len(self.__joint_list))
        return

    def paint_model(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        
        x = []
        y = []
        z = []
        for joint in self.__joint_list:
            pos = joint.get_global_pos()
            # pos = joint.get_local_pos()
            print(pos)
            x.append(pos[0])
            y.append(pos[1])
            z.append(pos[2])
        ax.scatter3D(x, y, z)    
        plt.show()
        return

if __name__ =="__main__":
    ori_file = os.path.join(os.getcwd(), ori_file)
    with open(ori_file, 'r') as myfile:
        data=myfile.read()

    # parse file
    obj = json.loads(data)

    KEY = ["Skeleton", "BodyDefs", "DrawShapeDefs"]
    # build model
    joint_json_list = obj[KEY[0]]["Joints"]
    joint_model = Model()
    joint_model.load_model(joint_json_list = joint_json_list)
    joint_model.paint_model()
from archive.motion_squeezer_for_skeleton import Joint
import os
import json
import numpy as np


class Model(object):
    def __init__(self, path):
        assert os.path.exists(path)
        self.load_model(path)
        print(f"[log] load model {path} succ")

    def load_model(self, path):
        self.joint_lst = []
        with open(path, "r") as f:
            json_cont = json.load(f)
            joints = json_cont["Skeleton"]["Joints"]
            for single_joint in joints:
                # print(single_joint)
                id = single_joint["ID"]
                name = single_joint["Name"]
                joint_type = single_joint["Type"]

                # print(f"{id} {name} {joint_type}")
                self.joint_lst.append(Joint(int(id), name, joint_type))
        self.joint_lst.sort(key=lambda x: x.id)
        self.__set_action_dof_offset()
        self.__set_motion_dof_offset()
        self.__set_action_dof_offset()

    def __set_freedom_offset(self):
        self.num_of_freedom = 0
        for single_joint in self.joint_lst:
            single_joint.freedom_offset = self.num_of_freedom
            self.num_of_freedom += single_joint.get_joint_num_of_freedom()

    def __set_motion_dof_offset(self):
        self.motion_size = 1             # the first number is motion phase
        for single_joint in self.joint_lst:
            single_joint.motion_offset = self.motion_size
            self.motion_size += single_joint.get_joint_motion_dof()

    def __set_action_dof_offset(self):
        self.action_size = 0
        for single_joint in self.joint_lst:
            single_joint.action_offset = self.action_size
            self.action_size += single_joint.get_joint_action_dof()

    def get_joint_by_id(self, id: int):
        assert id < len(self.joint_lst)
        return self.joint_lst[id]

    def get_joint_by_name(self, name: str):
        for i in self.joint_lst:
            if i.name == name:
                return i
        assert False, f"no joint name {name} in {[i.name for i in self.joint_lst]}"

    def get_num_of_joint(self):
        return len(self.joint_lst)


class Motion(object):
    def __init__(self, path, char: Model):
        assert os.path.exists(path)
        with open(path) as f:
            self.cont = json.load(f)

        self.frames = self.cont["Frames"]
        self.num_of_frames = len(self.frames)
        self.__check_motion(char)

    def __check_motion(self, char: Model):
        for i in self.frames:
            assert len(i) == char.motion_size

    def get_motion_by_frameid(self, frameid):
        return self.frames[frameid]

    def set_motion_by_frameid(self, motion, id):
        self.frames[id] = motion

    def save_motion(self, path):
        self.cont["Frames"] = self.frames
        with open(path, 'w') as f:
            json.dump(self.cont, f)


def joint_motion_conventer(old_joint: Joint, new_joint: Joint, old_motion_seg: list):
    '''
        Given the old joint and new joint, also with the old motion segment
        calcualte the new motion for given new joint

        if the type(old_joint) == type(new_joint), doesn't do any convent
    '''
    from math_util import quaterniont_to_aa
    print(f"old joint {old_joint.name} motion {old_motion_seg}")
    new_motion = []
    if (old_joint.type == Joint.JointType.SPHERICAL) and (new_joint.type == Joint.JointType.REVOLUTE):
        # spherical to revolute, extract the x component in aa
        aa = quaterniont_to_aa(np.array(old_motion_seg)).tolist()
        new_motion.append(aa[0])

    elif (old_joint.type == Joint.JointType.NONE) and (new_joint.type == Joint.JointType.BIPEDAL_NONE):
        # None to bipedal none, extract the YOZ translation and x rotation

        x_rot = quaterniont_to_aa(old_motion_seg[3:])[0]
        y_transl = old_motion_seg[1]
        z_transl = old_motion_seg[2]
        new_motion = [y_transl, z_transl, x_rot]

    else:
        if old_joint.type == new_joint.type:
            new_motion = old_motion_seg
        else:
            raise ValueError(
                f"unsupported motion convention from {old_joint.type} to {new_joint.type}")
    return new_motion


def convert_to_new_motion(old_model: Model, new_model: Model, old_motion_frame: list):
    num_of_joints = old_model.get_num_of_joint()
    assert new_model.get_num_of_joint() == num_of_joints
    assert ([i.name for i in old_model.joint_lst] == [
            i.name for i in new_model.joint_lst])

    new_motion_frame = [old_motion_frame[0]]
    for j_id in range(old_model.get_num_of_joint()):
        old_joint = old_model.get_joint_by_id(j_id)
        new_joint = new_model.get_joint_by_id(j_id)

        new_motion_frame += joint_motion_conventer(
            old_joint, new_joint, old_motion_frame[old_joint.motion_offset: old_joint.motion_offset + old_joint.get_joint_motion_dof()])

    return new_motion_frame


proj_dir = "/home/xudong/Projects/DeepMimic/"
skeleton_3d_path = "data/0908/characters/skeleton_legs.json"
motion_3d_path = "data/0908/motions/walk_motion_legs.txt"
skeleton_2d_path = "data/0111/characters/skeleton_bipedal_legs.json"
if __name__ == "__main__":

    # 1. load models and motion

    model_3d = Model(os.path.join(proj_dir, skeleton_3d_path))
    model_2d = Model(os.path.join(proj_dir, skeleton_2d_path))
    motion = Motion(os.path.join(proj_dir, motion_3d_path), model_3d)

    # 2. build the map and define the convention
    for id in range(motion.num_of_frames):
        new_motion_frame = convert_to_new_motion(
            model_3d, model_2d, motion.get_motion_by_frameid(id))
        motion.set_motion_by_frameid(new_motion_frame, id)

    motion.save_motion("2d_motion.json")
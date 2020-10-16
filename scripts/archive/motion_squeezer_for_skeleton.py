import os
import json
import csv
import sys
from enum import IntEnum
import random


project_dir = "/Users/xudong/Projects/DeepMimic"
old_skeleton_path = "../data/0805/characters/skeleton_080501.json"
new_skeleton_path = "../data/0908/characters/skeleton_legs.json"
old_motion_path = "../data/0426/motions/walk_motion_042602.txt"
new_motion_path = "../data/0908/motions/walk_motion_legs.txt"


class Joint(object):
    class JointType(IntEnum):
        NONE = 0
        REVOLUTE = 1
        SPHERICAL = 2
        FIXED = 3

    JOINT_TYPE_STR = ["none", "revolute", "spherical", "fixed"]
    JOINT_DOF = [7, 1, 4, 0]

    @staticmethod
    def decide_joint_type(joint_type_str):

        for joint_type, j_str in enumerate(Joint.JOINT_TYPE_STR):
            if j_str == joint_type_str:
                return Joint.JointType(joint_type)

        assert False, f"joint type str {joint_type_str} invalid"

    @staticmethod
    def decide_joint_dof(joint_type: JointType) -> int:
        assert isinstance(joint_type, Joint.JointType)
        return Joint.JOINT_DOF[int(joint_type)]

    def __init__(self, joint_id_, joint_name, joint_type_str):
        assert isinstance(joint_id_, int)
        self.joint_id = joint_id_
        self.joint_name = joint_name
        self.joint_type = Joint.decide_joint_type(joint_type_str)
        self.joint_dof = Joint.decide_joint_dof(self.joint_type)

    def get_joint_type(self):
        return self.joint_type

    def get_joint_name(self):
        return self.joint_name

    def get_joint_dof(self):
        return self.joint_dof

    def get_joint_id(self):
        return self.joint_id

    def set_offset(self, offset):
        self.offset = offset

    def get_offset(self):
        return self.offset


def find_joint_by_name(joint_name, joint_lst):
    assert all(
        isinstance(item, Joint) for item in joint_lst
    ), "joint_lst joints is not Joint"
    for single_joint in joint_lst:
        if joint_name == single_joint.get_joint_name():
            return single_joint.get_joint_id()
    raise ValueError(f"fail to get the joint name {joint_name}")


def find_joint_by_id(joint_id, joint_lst):
    assert all(
        isinstance(item, Joint) for item in joint_lst
    ), "joint_lst joints is not Joint"
    for single_joint in joint_lst:
        if joint_id == single_joint.get_joint_id():
            return single_joint
    raise ValueError(f"fail to get the joint id {joint_id}")


def get_skeleton_info(path):
    joint_lst = []
    with open(path, "r") as f:
        json_cont = json.load(f)
        joints = json_cont["Skeleton"]["Joints"]
        for single_joint in joints:
            # print(single_joint)
            id = single_joint["ID"]
            name = single_joint["Name"]
            joint_type = single_joint["Type"]

            print(f"{id} {name} {joint_type}")
            joint_lst.append(Joint(int(id), name, joint_type))
    joint_lst.sort(key=lambda x: x.get_joint_id())
    offset = 0
    for single_joint in joint_lst:
        single_joint.set_offset(offset)
        print(f"joint {single_joint.get_joint_name()} offset {offset}")
        offset += single_joint.get_joint_dof()

    return joint_lst


# confirm the new skeleton is consists by the old skeleton, and build a map for joint ids (in the order of new joints)


def build_joint_map_from_old_skeleton_to_new_skeleton(old_joints, new_joints):
    joint_id_map = {}
    assert all(
        isinstance(item, Joint) for item in new_joints
    ), "New joints is not Joint"
    assert all(isinstance(item, Joint)
               for item in old_joints), "Old joint is not Joint"

    for new_joint in new_joints:
        old_joint_id = find_joint_by_name(
            new_joint.get_joint_name(), old_joints)
        joint_id_map[new_joint.get_joint_id()] = old_joint_id
    return joint_id_map


def convert_one_frame(old_frame, joint_map, old_joint_array, new_joint_array):
    assert isinstance(joint_map, dict)
    timestep_offset = 1

    # add the timestep
    new_frame = [old_frame[0]]

    # add other stuffs
    for new_joint in new_joint_array:
        new_joint_id = new_joint.get_joint_id()
        old_joint_id = joint_map[new_joint_id]
        old_joint = old_joint_array[old_joint_id]
        old_joint_offset = old_joint.get_offset() + timestep_offset
        old_joint_dof = old_joint.get_joint_dof()

        new_joint = new_joint_array[new_joint_id]
        new_joint_offset = new_joint.get_offset() + timestep_offset
        new_joint_dof = new_joint.get_joint_dof()
        assert new_joint_dof == old_joint_dof, "the joint dof doesn't match"
        # print(
        #     f"new joint {new_joint_id} offset {new_joint_offset}, old joint {old_joint_id} offset {old_joint_offset}"
        # )

        new_frame += old_frame[old_joint_offset: old_joint_offset + old_joint_dof]
    return new_frame
    # print(f"new frame {new_frame}")


if __name__ == "__main__":
    old_joints = get_skeleton_info(old_skeleton_path)
    new_joints = get_skeleton_info(new_skeleton_path)
    joint_id_map_from_old_to_new = build_joint_map_from_old_skeleton_to_new_skeleton(
        old_joints, new_joints
    )

    # with open(old_motion_path, "r") as f_old:
    #     # new_motion_json = json.load(f_new)
    #     new_motion_json = json.load(f_old)

    #     for frame_id, single_frame in enumerate(new_motion_json["Frames"]):

    #         new_frame = convert_one_frame(
    #             single_frame, joint_id_map_from_old_to_new, old_joints, new_joints
    #         )
    #         new_motion_json["Frames"][frame_id] = new_frame
    #         print(f"cur new frame {new_frame}")
    #     with open(new_motion_path, "w") as f_new:
    #         json.dump(new_motion_json, f_new, indent=4)
    #         print(f" write new json to {new_motion_path}")
    #     # open(new_motion_path, "w") as f_new

import os
import json
import csv
import argparse

'''
    验证skeleton文件中, drawshapes和bodydefs是否相同?左右的参数是否对称?
'''

def verify_drawshapes_and_bodydefs(character_path):
    json_cont = None
    with open(character_path, "r") as f:
        json_cont = json.load(f)

    joints = json_cont["Skeleton"]["Joints"]
    bodydefs = json_cont["BodyDefs"]
    drawshapdefs = json_cont["DrawShapeDefs"]

    # 验证drawshapedefs中每一项都一样
    for i in bodydefs:
        name = i["Name"]
        for q in drawshapdefs:
            if q["Name"] == name:
                # 这样就匹配了, 开始遍历i的所有k
                for key in i.keys():
                    if key in q.keys():
                        try:
                            assert i[key] == q[key]
                            # print("in joint {},value {} {} = {} ".format(str(name), key, i[key], q[key]) )
                        except:
                            print("in joint %s, the value %s in drawdefs is %s, but in bodydefs is %s"
                                  %(name, key, str(q[key]), str(i[key])))
                            raise(ValueError)
    print("{} drawshapes = bodydefs, verified true".format(character_path))

def verify_symmetric(character_path):
    '''
        check 其中3项的对称情况:
        1. skeleton-joints
        2. drawshapes
        3. bodydefs
    '''
    name_root_lst = ["hipjoint", "femur", "tibia", "foot", "clavicle", "humerus", "radius", "wrist"]
    json_cont = None
    with open(character_path, "r") as f:
        json_cont = json.load(f)

    def verify_2_joints(scope, joint1_json, joint2_json, check_items):
        # 验证2个joint的json在check_items里面的项是否相同
        all_keys = joint1_json.keys()
        for key in all_keys:
            if key in check_items:# 如果key处在检查项中
                try:
                    v1 = joint1_json[key]
                    v2 = joint2_json[key]
                    assert type(v1) == type(v2)
                    if type(v1) == float:
                        v1,v2 = round(v1, 5), round(v2, 5)
                    assert v1 == v2
                except:
                    print("\t[error] scope %s joint %s and joint %s in %s is diff: %s and %s"
                    % (scope, joint1_json["Name"], joint2_json["Name"], key,
                    joint1_json[key], joint2_json[key]))
        return True

    def verify_skeleton_symmetric():
        # 验证bodydefs是不是对称的
        print("******************************************begin verify skeleton******************************************")
        joints = json_cont["Skeleton"]["Joints"]
        check_items = ["AttachThetaX",  "AttachThetaY", "AttachThetaZ", "AttachX",
            "AttachY", "AttachZ","DiffWeight", "IsEndEffector", "LimHigh0", "LimLow0",
            "LimHigh1", "LimLow1", "LimHigh2", "LimLow2","TorqueLim", "Type"]

        joints_dict = {}
        for i in joints:
            joints_dict[i["Name"]] = i
        
        for root in name_root_lst:
            print("[log] skeleton verify %s begin" % root)
            joint1 = joints_dict["r" + root]
            joint2 = joints_dict["l" + root]
            verify_2_joints("skeleton", joint1, joint2, check_items)
            

    def verify_bodydefs_symmetric():
        print("******************************************\
begin verify bodydefs\
*******************************************")

        joints = json_cont["BodyDefs"]
        check_items = ["AttachThetaX", "AttachThetaY", "AttachThetaZ",
            "AttachX", "AttachY", "AttachZ", "BottomScale","ColGroup", 
            "ColorA", "ColorB", "ColorG", "ColorR", "EnableFallContact", 
            "Mass", "Param0", "Param1", "Param2", "Shape", "TopScale"]
        joints_dict = {}
        for i in joints:
            joints_dict[i["Name"]] = i
        for root in name_root_lst:
            print("[log] bodydefs verify %s begin" % root)
            joint1 = joints_dict["r" + root]
            joint2 = joints_dict["l" + root]
            verify_2_joints("bodydefs", joint1, joint2, check_items)
    verify_skeleton_symmetric()
    verify_bodydefs_symmetric()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str, default=".")
    arg = parser.parse_args()
    character_dir = os.path.join(os.getcwd(), arg.dir)
    files = os.listdir(character_dir)
    for i in files:
        print("[log] begin verify %s" % i)
        if i.find(".json") != -1 and i.find("skeleton") != -1 or i.find("body") !=-1:
            verify_drawshapes_and_bodydefs(os.path.join(character_dir, i))
            # verify_symmetric(os.path.join(character_dir, i))
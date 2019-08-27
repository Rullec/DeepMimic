import os
import json
import csv
import argparse

'''
    验证skeleton文件中, drawshapes和bodydefs是否相同
'''

def verify(character_path):
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
    print("{} verified true".format(character_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str, default=".")
    arg = parser.parse_args()
    character_dir = arg.dir
    files = os.listdir(character_dir)
    for i in files:
        if i.find(".json") != -1 and i.find("skeleton") != -1:
            verify(os.path.join(character_dir, i))
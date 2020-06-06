import numpy as np
import json
import os

st_num = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 46, 50, 55]
origin_path_dir = "../data/paths"
target_path_dir = "../data/paths_normalized"
files = os.listdir(origin_path_dir)

# use to normlize the joint axis angle norm. given a uniform norm
# normalize_goal = 0.627
normalize_goal = 1

do_not_normalized = False
# use to normlize the joier axis angle norm. given a per-joint norm
# normalize_goal_per_joint = [ normalize_goal + np.random.normal(scale=0.1) for i in range(len(st_num))]
# assert(min(normalize_goal_per_joint)>0)

# print(normalize_goal_per_joint)
# exit(0)
def target_normalized(number, normalize_lst):
    assert(type(number) == list)
    assert(len(normalize_lst) > 0)
    global normalize_goal
    new_number = np.array(number, dtype=np.float)
    if do_not_normalized is True:
        if np.random.randint(low = 0, high = 100) == 1:
            print("do not noramlized!")
        return new_number.tolist()

    for id, i in enumerate(normalize_lst):
        # add sclae = 0.5 gaussian noise
        # normalized_seg = new_number[i:i+3] * 1.0 / np.linalg.norm(new_number[i:i+3]) * (normalize_goal + np.random.normal(scale=0.05)

        # no noise
        normalized_seg = new_number[i:i+3] * 1.0 / np.linalg.norm(new_number[i:i+3]) * normalize_goal

        # per joint noise, but very smooth
        # normalized_seg = new_number[i:i+3] * 1.0 / np.linalg.norm(new_number[i:i+3]) * (normalize_goal_per_joint[id])

        # per joint noise and a little uniform noise
        # normalized_seg = new_number[i:i+3] * 1.0 / np.linalg.norm(new_number[i:i+3]) * (normalize_goal_per_joint[id] + (np.random.uniform()-0.1) * 0.3)
        # if np.linalg.norm(normalized_seg) < 0:
        #     print(normalized_seg)
        #     print(normalize_goal_per_joint[id])
        #     exit(0)
        
        # import numpy as np
        # [(np.random.uniform() + 1) * 0.02 for i in range(10)]
        new_number[i] = normalized_seg[0]
        new_number[i + 1] = normalized_seg[1]
        new_number[i + 2] = normalized_seg[2]
    return new_number.tolist()

def NormalizeAction(filename, target_filename):
    assert(os.path.exists(filename))
    print("begin to handle %s" % filename)
    try:
        with open(filename, 'r') as f:
            root = json.load(f)
    except :
        print("load %s failed" % filename)
        return 
    actions = root["actions"]
    new_actions = []
    for single_action in actions:
        new_actions.append(target_normalized(single_action, st_num)) 
    assert(len(new_actions) == len(actions))

    root["actions"] = new_actions
    with open(target_filename, 'w') as f:
        json.dump(root, f)
        print("write json to %s" % target_filename)
    return 

import shutil
from multiprocessing import Pool
if __name__ == "__main__":    
    # numbers = [float(i) for i in range(100)]
    # new_numbers = target_normalized(numbers, st_num)
    # print("before %s" % str(numbers))
    # print("after %s" % str(new_numbers))
    if os.path.exists(target_path_dir):
        shutil.rmtree(target_path_dir)
    os.makedirs(target_path_dir)
    arts = []
    for file in files:
        if -1 != file.find("json"):
            arts.append([os.path.join(origin_path_dir, file), os.path.join(target_path_dir, file)])
            # NormalizeAction(os.path.join(origin_path_dir, file), os.path.join(target_path_dir, file))
    pool = Pool(12)
    pool.starmap(NormalizeAction, arts)
    

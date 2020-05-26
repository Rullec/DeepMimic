# convert all axis angle theta into positive
import os
import json
import numpy as np
st_num = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 46, 50, 55]
origin_path_dir = "./paths_normalized"
target_path_dir = "./paths_normalized_positived"
files = os.listdir(origin_path_dir)

def action_positived(single_action, st_lst):
    for axis_st_id in st_lst:
        raw_data = single_action[axis_st_id-1: axis_st_id+3]
        if raw_data[0] < 0:
            single_action[axis_st_id-1 : axis_st_id + 3] = (-np.array(raw_data)).tolist()

            # print("from old %s to new %s" % (raw_data, single_action[axis_st_id-1 : axis_st_id + 3]))
    return single_action

def PositiveAction(filename, target_filename):
    assert(os.path.exists(filename))
    print("begin to handle %s" % filename)
    with open(filename, 'r') as f:
        root = json.load(f)
    actions = root["actions"]
    new_actions = []
    for single_action in actions:
        new_actions.append(action_positived(single_action, st_num)) 
    assert(len(new_actions) == len(actions))

    root["actions"] = new_actions
    with open(target_filename, 'w') as f:
        json.dump(root, f)
        print("write json to %s" % target_filename)
    return 

if __name__ == "__main__":    
    for file in files:
        if -1 != file.find("json"):
            PositiveAction(os.path.join(origin_path_dir, file), os.path.join(target_path_dir, file))

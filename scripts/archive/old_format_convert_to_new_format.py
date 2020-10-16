import json
import numpy as np
import os
from multiprocessing import Pool


def convert_old_path_to_new_traj(old_data_filename, new_data_filename):
    print("from %s to %s" % (old_data_filename, new_data_filename))
    try:
        # 1. load old data
        with open(old_data_filename, 'r') as f:
            old_root = json.load(f)

        # 2. write new data to disk
        num_of_frame = len(old_root["states"]) - 1
        new_root = {}
        new_root["num_of_frames"] = num_of_frame
        new_root["data_list"] = []
        for frame_id in range(num_of_frame):
            new_single_frame = {}
            new_single_frame["action"] = old_root["actions"][frame_id]
            new_single_frame["state"] = old_root["states"][frame_id]
            new_single_frame["reward"] = old_root["rewards"][frame_id]
            new_single_frame["frame_id"] = frame_id
            new_root["data_list"].append(new_single_frame)

        with open(new_data_filename, 'w') as f:
            json.dump(new_root, f)
    except:
        print("[error] %s converted faile", old_data_filename)
    return 0


def output_summary_table(table_filename, old_data_dir, new_data_dir):
    root = {}
    root["char_file"] = "data/0424/characters/skeleton_042302_revised.json"
    root["controller_file"] = "data/0424/controllers/humanoid3d_ctrl_skeleton_0424.txt"
    root["num_of_trajs"] = 1200

    old_files = os.listdir(old_data_dir)
    root["single_trajs_lst"] = []

    mpi_params_lst = []
    for cur_old_filename in old_files:
        if -1 != cur_old_filename.find("json"):
            cur_new_filename = os.path.join(new_data_dir, cur_old_filename)
            cur_entry = {}
            cur_entry["length_second"] = 2.0
            cur_entry["num_of_frame"] = 60
            cur_entry["train_data_filename"] = cur_new_filename
            root["single_trajs_lst"].append(cur_entry)
            mpi_params_lst.append(
                (os.path.join(old_data_dir, cur_old_filename), cur_new_filename))

        # convert_old_path_to_new_traj()

    pool = Pool(12)
    pool.starmap(convert_old_path_to_new_traj, mpi_params_lst)

    with open(table_filename, 'w') as f:
        json.dump(root, f)
        print("summary table write to %s" % table_filename)

    return 0


old_format_data_dir = "/home/xudong/Projects/DeepMimic/data/paths_normalized/"
new_format_data_dir = "/home/xudong/Projects/DeepMimic/data/batch_train_data/path_converted"

if __name__ == "__main__":
    assert(os.path.exists(old_format_data_dir)
           and os.path.exists(new_format_data_dir))
    summary_table_path = os.path.join(
        new_format_data_dir, "summary_table.json")
    print(summary_table_path)
    output_summary_table(summary_table_path,
                         old_format_data_dir, new_format_data_dir)

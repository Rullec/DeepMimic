import os
import json


class param:
    def __init__(self, short_name, full_key_name, value_lst):
        assert type(short_name) == str
        assert type(full_key_name) == str
        assert type(value_lst) == list
        self.short_name = short_name
        self.full_key = full_key_name
        self.v_lst = value_lst
        print(
            f"[log] create param {short_name} for {full_key_name} list {value_lst}")


def generate_new_agent(standard_filename, new_filename, key_value_dict):
    with open(standard_filename) as f:
        cont = json.load(f)
    for key in key_value_dict.keys():
        assert key in cont
        cont[key] = key_value_dict[key]

    with open(new_filename, "w") as f:
        json.dump(cont, f)
    print(f"[log] write down {new_filename} succ")


lr_param = param("lr", "LearningRate", [1e-2, 1e-3, 1e-4])
lr_decay_param = param("lr_decay", "LearningRateDecay", [0.999])
buffer_size_param = param(
    "replay_size", "ReplayBufferCapacity", [100, 300, 500])


standard_agent_file = "agent_diffstd.json"
pure_agent_name = standard_agent_file[:standard_agent_file.find(".")]
output_dir = "time1/"
if os.path.exists(output_dir) is False:
    os.makedirs(output_dir)

for lr in lr_param.v_lst:
    for lr_decay in lr_decay_param.v_lst:
        for buffer_size in buffer_size_param.v_lst:

            full_name = f"{pure_agent_name}" + f"_{lr_param.short_name}_{lr}" + \
                f"_{lr_decay_param.short_name}_{lr_decay}" + \
                f"_{buffer_size_param.short_name}_{buffer_size}" + ".json"
            full_name = os.path.join(output_dir, full_name)
            generate_new_agent(
                standard_agent_file,
                full_name,
                {
                    lr_param.full_key: lr,
                    lr_decay_param.full_key: lr_decay,
                    buffer_size_param.full_key: buffer_size,
                }
            )

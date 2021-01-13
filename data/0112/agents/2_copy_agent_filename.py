import os

agent_dir = "./time1"
output_dir = "exp_train_files/"
if os.path.exists(output_dir):
    assert False
if os.path.exists(output_dir) is False:
    os.makedirs(output_dir)

file_lst = os.listdir(agent_dir)

stand_train_file = "../../../args/0112/train_diff_legs_std.txt"
assert os.path.exists(stand_train_file)

for agent_file in file_lst:
    new_name = os.path.split(stand_train_file)[-1][:-4] + agent_file + ".txt"
    with open(stand_train_file) as f:
        cont = f.readlines()
        for idx, line in enumerate(cont):
            if line.find("--agent_files") != -1:
                cont[idx] = f"--agent_files data/0112/agents/time1/{agent_file}\n"

    output_newname = os.path.join(output_dir, new_name)
    with open(output_newname, 'w') as f:
        f.writelines(cont)
    print(f"[log] write {new_name} succ")
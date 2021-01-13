import os
train_files_dir = "../../../args/0112/time1"
assert os.path.exists(train_files_dir)
files = os.listdir(train_files_dir)

for i in files:
    new_command = f"python -u DeepMimicTorch_Optimizer.py --arg_file args/0112/time1/{i} 1>logs/{i}.log 2>logs/{i}.err &"
    print(new_command)
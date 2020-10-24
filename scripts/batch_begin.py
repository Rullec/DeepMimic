import os
kernel_num = 7
arg_file_lst = [
    # "args/1024/100hz_train.txt",
    # "args/1024/60hz_train.txt",
    "args/1024/300hz_train.txt",
    "args/1024/600hz_train.txt"
]


cmd_lst = []
# set up the log and error
for arg_file_path in arg_file_lst:
    arg_file = os.path.split(arg_file_path)[-1]
    assert(type(arg_file) is str)
    arg_file_prefix = arg_file[:-4]
    print(arg_file_prefix)
    log_file = f"logs/{arg_file_prefix}.log"
    err_file = f"logs/{arg_file_prefix}.err"
    if os.path.exists(log_file) == True:
        assert False, f"log file {log_file} exists and should not be overwrited"
    cmd = f"(mpiexec -n {kernel_num} python DeepMimic_Optimizer.py --arg_file {arg_file_path} 1>{log_file} 2>{err_file} &)"
    cmd_lst.append(cmd)
    print(f"arg file {arg_file_path}\ncmd {cmd}")
    os.popen(cmd)

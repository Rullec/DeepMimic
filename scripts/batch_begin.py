import os
kernel_num = 2
arg_file_lst = [
    "args/1019/param_search/search_0.5_16.txt",
    "args/1019/param_search/search_0.5_24.txt",
    "args/1019/param_search/search_0.5_32.txt",
    "args/1019/param_search/search_1_16.txt",
    "args/1019/param_search/search_1_24.txt",
    "args/1019/param_search/search_1_32.txt"]

cmd_lst = []
# set up the log and error
for arg_file_path in arg_file_lst:
    arg_file =  os.path.split(arg_file_path)[-1]
    assert(type(arg_file) is str)
    arg_file_prefix = arg_file[:-4]
    print(arg_file_prefix)
    log_file = f"logs/{arg_file_prefix}.log"
    err_file = f"logs/{arg_file_prefix}.err"

    cmd = f"(mpiexec -n {kernel_num} python DeepMimic_Optimizer.py --arg_file {arg_file_path} 1>{log_file} 2>{err_file} &)"
    cmd_lst.append(cmd)
    print(f"arg file {arg_file_path}\ncmd {cmd}")
    os.popen(cmd)


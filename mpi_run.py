import sys
import subprocess
from util.arg_parser import ArgParser
from util.logger import Logger

'''
这个函数就是最终训练agent的函数

疑问:
1. worker数量对应的是什么概念?

'''
def main():
    # Command line argument
    args = sys.argv[1:]
    # 这个argv是"['/home/fengxudong/Projects/deepmimic/mpi_run.py', '--arg_file', 'args/train_humanoid3d_run_args.txt', '--num_workers', '1']"
    # 然后argv[1:]就是去除了文件名以后的部分

    arg_parser = ArgParser()
    arg_parser.load_args(args)

    num_workers = arg_parser.parse_int('num_workers', 1)    # 解析获得worker数目(虽然还不知道worker是干啥的)
    assert(num_workers > 0)

    Logger.print('Running with {:d} workers'.format(num_workers))
    cmd = 'mpiexec -n {:d} python DeepMimic_Optimizer.py '.format(num_workers)
    cmd += ' '.join(args)
    Logger.print('cmd: ' + cmd)

    # 调用子进程执行这句话:
    # 打开另一个进程执行Deepmimic_Optimizer.py,
    '''mpiexec -n 1 python DeepMimic_Optimizer.py --arg_file args/train_humanoid3d_run_args.txt --num_workers 1'''
    subprocess.call(cmd, shell=True)
    return

if __name__ == '__main__':
    main()

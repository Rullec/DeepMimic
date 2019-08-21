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

    arg_parser = ArgParser()
    arg_parser.load_args(args)

    num_workers = arg_parser.parse_int('num_workers', 12)    # mpiexec中并行几个进程?
    assert(num_workers > 0)
    
    Logger.print('Running with {:d} workers'.format(num_workers))
    cmd = 'mpiexec -n {:d} python DeepMimic_Optimizer.py '.format(num_workers)
    cmd += ' '.join(args)
    Logger.print('cmd: ' + cmd)

    subprocess.call(cmd, shell=True)
    return

if __name__ == '__main__':
    # print("参数是:  %s " % str(sys.argv[1:]))
    main()
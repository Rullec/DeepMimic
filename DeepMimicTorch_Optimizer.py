import numpy as np
import sys
from env.deepmimic_env import DeepMimicEnv
from learning.torch.rl_world_torch import RLWorldTorch
from util.logger import Logger
from DeepMimicTorch import update_world, update_timestep, build_world
import util.mpi_util as MPIUtil
import time

args = []
world = None


def run():
    global update_timestep
    global world

    done = False
    while not (done):
        update_world(world, update_timestep)
    return


def shutdown():
    global world

    Logger.print('Shutting down...')
    world.shutdown()
    return


def main():
    global args
    global world

    # Command line arguments
    args = sys.argv[1:]
    world = build_world(args, enable_draw=False)

    run()

    shutdown()

    return


if __name__ == '__main__':
    import torch
    import util.mpi_util as MPIUtil
    import numpy as np
    torch.manual_seed(MPIUtil.get_proc_rank())
    np.random.seed(MPIUtil.get_proc_rank())
    # torch.manual_seed(0)
    # np.random.seed(0)
    main()

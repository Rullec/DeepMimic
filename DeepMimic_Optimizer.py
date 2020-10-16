import numpy as np
import sys
from env.deepmimic_env import DeepMimicEnv
from learning.rl_world import RLWorld
from util.logger import Logger
from DeepMimic import update_world, update_timestep, build_world
import util.mpi_util as MPIUtil
import time

args = []
world = None


def run():
    global update_timestep
    global world

    done = False
    # print("*******************begin main loop")
    while not (done):
        # st = time.time()
        # 无限循环进行update_world
        update_world(world, update_timestep)
        # break
        # ed = time.time()
        # print("[log] Optimizer - main - run - update world done, timestep = %.3f, cost time = %.3f" % (update_timestep, ed - st))
    # print("*******************end main loop")
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

    # print(args)

    # print("[log] Deepmimic_Optimizer.py : begin build world")
    # world = build_world(args, enable_draw=False)
    world = build_world(args, enable_draw=False)
    # print("[log] Deepmimic_Optimizer.py : build world done")

    run()

    shutdown()

    return


if __name__ == '__main__':
    main()

'''
     --arg_file args/run_humanoid3d_spinkick_args.txt
'''
import numpy as np
import os
import sys
import random
import platform as os_pt

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from env.deepmimic_env import DeepMimicEnv
from learning.rl_world import RLWorld
from util.arg_parser import ArgParser
from util.logger import Logger
import util.mpi_util as MPIUtil
import util.util as Util

# Dimensions of the window we are drawing into.
win_width = 800
win_height = int(win_width * 9.0 / 16.0)
reshaping = False

# anim
fps = 60
update_timestep = 1.0 / fps
display_anim_time = int(1000 * update_timestep)
animating = True

playback_speed = 1
playback_delta = 0.05

# FPS counter
prev_time = 0
updates_per_sec = 0

args = []
world = None

def build_arg_parser(args):
    arg_parser = ArgParser()
    arg_parser.load_args(args)

    arg_file = arg_parser.parse_string('arg_file', '')
    if (arg_file != ''):
        arg_file = os.path.realpath(os.path.join(os.getcwd(), arg_file))
        succ = arg_parser.load_file(arg_file)
        assert succ, Logger.print('Failed to load args from: ' + arg_file)

    rand_seed_key = 'rand_seed'
    if (arg_parser.has_key(rand_seed_key)):
        rand_seed = arg_parser.parse_int(rand_seed_key)
        rand_seed += 1000 * MPIUtil.get_proc_rank()
        Util.set_global_seeds(rand_seed)

    return arg_parser


def update_intermediate_buffer():
    if not (reshaping):
        if (win_width != world.env.get_win_width() or win_height != world.env.get_win_height()):
            world.env.reshape(win_width, win_height)

    return

def update_world(world, time_elapsed):
    '''
        主循环调用这个update函数
    '''
    # 一次update_world在实践中被分成很多个子步
    # 这可能是为了提高帧率或者实现一些别的功能?
    num_substeps = world.env.get_num_update_substeps()
    timestep = time_elapsed / num_substeps
    num_substeps = 1 if (time_elapsed == 0) else num_substeps

    for i in range(num_substeps):
        world.update(timestep)
        
        valid_episode = world.env.check_valid_episode()

        if valid_episode:   # 速度没有爆炸
            end_episode = world.env.is_episode_end()
            if (end_episode):
                # print("episode done")
                world.end_episode()
                world.reset()
                break
        else:   # 速度爆炸了
            # 一旦出现无效episode，直接就world.reset了。
            world.reset()
            break
    return

def draw():
    global reshaping

    update_intermediate_buffer()
    world.env.draw()
    
    glutSwapBuffers()
    reshaping = False

    return

def reshape(w, h):
    global reshaping
    global win_width
    global win_height

    reshaping = True
    win_width = w
    win_height = h

    return

def step_anim(timestep):
    global animating
    global world

    update_world(world, timestep)
    animating = False
    glutPostRedisplay()
    return

def reload():
    global world
    global args

    # 在这里build了world, 必然绘制
    world = build_world(args, enable_draw=True)
    # print("build world succ")
    return

def reset():
    world.reset()
    return

def get_num_timesteps():
    global playback_speed

    num_steps = int(playback_speed)
    if (num_steps == 0):
        num_steps = 1

    num_steps = np.abs(num_steps)
    return num_steps

def calc_display_anim_time(num_timestes):
    global display_anim_time
    global playback_speed

    anim_time = int(display_anim_time * num_timestes / playback_speed)
    anim_time = np.abs(anim_time)
    return anim_time

def shutdown():
    global world

    Logger.print('Shutting down...')
    world.shutdown()
    sys.exit(0)
    return

def get_curr_time():
    curr_time = glutGet(GLUT_ELAPSED_TIME)
    return curr_time

def init_time():
    global prev_time
    global updates_per_sec
    prev_time = get_curr_time()
    updates_per_sec = 0
    return

def animate(callback_val):
    global prev_time
    global updates_per_sec
    global world

    counter_decay = 0

    if (animating):
        num_steps = get_num_timesteps()
        curr_time = get_curr_time()
        time_elapsed = curr_time - prev_time
        prev_time = curr_time;

        timestep = -update_timestep if (playback_speed < 0) else update_timestep
        for i in range(num_steps):
            update_world(world, timestep)
        
        # FPS counting
        update_count = num_steps / (0.001 * time_elapsed)
        if (np.isfinite(update_count)):
            updates_per_sec = counter_decay * updates_per_sec + (1 - counter_decay) * update_count;
            world.env.set_updates_per_sec(updates_per_sec);
            
        timer_step = calc_display_anim_time(num_steps)
        update_dur = get_curr_time() - curr_time
        timer_step -= update_dur
        timer_step = np.maximum(timer_step, 0)
        
        glutTimerFunc(int(timer_step), animate, 0)
        glutPostRedisplay()

    if (world.env.is_done()):
        shutdown()

    return

def toggle_animate():
    global animating

    animating = not animating
    if (animating):
        glutTimerFunc(display_anim_time, animate, 0)

    return

def change_playback_speed(delta):
    global playback_speed

    prev_playback = playback_speed
    playback_speed += delta
    world.env.set_playback_speed(playback_speed)

    if (np.abs(prev_playback) < 0.0001 and np.abs(playback_speed) > 0.0001):
        glutTimerFunc(display_anim_time, animate, 0)

    return

def toggle_training():
    global world

    world.enable_training = not world.enable_training
    if (world.enable_training):
        Logger.print('Training enabled')
    else:
        Logger.print('Training disabled')
    return

def keyboard(key, x, y):
    key_val = int.from_bytes(key, byteorder='big')
    world.env.keyboard(key_val, x, y)

    if (key == b'\x1b'): # escape
        shutdown()
    elif (key == b' '):
        toggle_animate();
    elif (key == b'>'):
        step_anim(update_timestep);
    elif (key == b'<'):
        step_anim(-update_timestep);
    elif (key == b','):
        change_playback_speed(-playback_delta);
    elif (key == b'.'):
        change_playback_speed(playback_delta);
    elif (key == b'/'):
        change_playback_speed(-playback_speed + 1);
    elif (key == b'l'):
        reload();
    elif (key == b'r'):
        reset();
    elif (key == b't'):
        toggle_training()

    glutPostRedisplay()
    return

def mouse_click(button, state, x, y):
    world.env.mouse_click(button, state, x, y)
    glutPostRedisplay()

def mouse_move(x, y):
    world.env.mouse_move(x, y)
    glutPostRedisplay()
    
    return

def init_draw():
    glutInit()  

    if os_pt.system() == "Darwin":
        glutInitDisplayMode(GLUT_3_2_CORE_PROFILE| GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    else:
        glutInitContextVersion(3, 2)
        glutInitContextFlags(GLUT_FORWARD_COMPATIBLE)
        glutInitContextProfile(GLUT_CORE_PROFILE)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

    glutInitWindowSize(win_width, win_height)
    glutCreateWindow(b'DeepMimic')
    return
    
def setup_draw():
    glutDisplayFunc(draw)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse_click)
    glutMotionFunc(mouse_move)
    glutTimerFunc(display_anim_time, animate, 0)

    reshape(win_width, win_height)
    world.env.reshape(win_width, win_height)
    
    return

def build_world(args, enable_draw, playback_speed=1):
    '''
        建造世界，这是第一个被调用的函数
    :param args:    --arg_file args/train_humanoid3d_run_args.txt --num_workers 1
    :param enable_draw: 在train下是一个false，别的不知道
    :param playback_speed: 播放速度默认是1
    :return:
    '''

    arg_parser = build_arg_parser(args)     # 参数解析器，之前已经看过，就是一个dict(value - list )而已
    env = DeepMimicEnv(args, enable_draw)   # 先创建env
    world = RLWorld(env, arg_parser)        # 然后在创建world, (创建完world再创建agent)
    world.env.set_playback_speed(playback_speed)
    # 为什么环境总是要被先创建?因为agent是依赖world中的env才能给出维度信息的，才能act的。world是舞台
    # 先有环境再有人
    return world

def draw_main_loop():
    init_time()
    glutMainLoop()
    return

def main():
    global args

    # Command line arguments
    args = sys.argv[1:]
    init_draw()
    reload()
    setup_draw()
    draw_main_loop()

    return

if __name__ == '__main__':
    # 如果调用Deepmimic.py的话，就会进行绘制...
    main()
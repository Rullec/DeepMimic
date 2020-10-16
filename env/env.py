from abc import ABC, abstractmethod
import numpy as np
from enum import Enum
from learning.normalizer import Normalizer


class Env(ABC):
    '''
        deepmimic中的环境类，就是从这里继承下去的
    '''
    class Terminate(Enum):   # 终止态三种case: 枚举
        Null = 0
        Fail = 1
        Succ = 2

    def __init__(self, args, enable_draw):
        self.enable_draw = enable_draw  # 是否启用绘制?
        return

    @abstractmethod
    def update(self, timestep):  # 更新update函数
        pass

    @abstractmethod
    def reset(self):            # reset，重置
        pass

    @abstractmethod
    def get_time(self):         # 获取当前时间，get time(应该就是积累的时间步长)
        pass

    @abstractmethod
    def get_name(self):         # 获取名字
        pass

    # rendering and UI interface
    def draw(self):             # 绘制和Ui的接口
        pass

    def keyboard(self, key, x, y):  # 键盘信号接受，在ENV类中
        pass

    def mouse_click(self, button, state, x, y):  # 鼠标点击
        pass

    def mouse_move(self, x, y):  # 鼠标移动
        pass

    def reshape(self, w, h):    # 变形，这些都是窗口操作
        pass

    def shutdown(self):         # 关闭窗口
        pass

    def is_done(self):
        return False

    def set_playback_speed(self, speed):    # 设置playback速度?什么意思?
        pass

    def set_updates_per_sec(self, updates_per_sec):  # 设置帧率?
        pass

    @abstractmethod
    def get_win_width(self):
        pass

    @abstractmethod
    def get_win_height(self):           # 获取窗口信息
        pass

    # 这个什么意思? num update substeps, substeps是什么意思?
    def get_num_update_substeps(self):
        return 1

    # rl interface

    @abstractmethod
    def is_rl_scene(self):  # rl 环境接口，看来他也没有完全的把环境和算法且分开
        return False

    @abstractmethod
    def get_num_agents(self):   # agent数量s
        return 0

    @abstractmethod
    def need_new_action(self, agent_id):
        '''
            给定一个agent 编号，判断是否需要新的action
        :param agent_id:
        :return:
        '''
        return False

    @abstractmethod
    def record_state(self, agent_id):
        '''
            记录状态
        :param agent_id:
        :return:
        '''
        pass

    @abstractmethod
    def record_goal(self, agent_id):
        '''
            记录目标o: 这里的目标是什么意思?
        :param agent_id:
        :return:
        '''
        pass

    @abstractmethod
    def set_action(self, agent_id):
        '''
            设置行动(action),也就是给agent动作指令
        :param agent_id:
        :return:
        '''
        pass

    @abstractmethod
    def get_action_space(self, agent_id):
        '''
            设置行为空间,专门针对某个agent得到action space?是否指的是活动空间大小。
                多agent在env层面上就实现了。
        :param agent_id:
        :return:
        '''
        pass

    @abstractmethod
    def get_state_size(self, agent_id):
        pass

    @abstractmethod
    def get_goal_size(self, agent_id):
        pass

    @abstractmethod
    def get_action_size(self, agent_id):
        pass

    @abstractmethod
    def get_num_actions(self, agent_id):
        pass

    @abstractmethod
    def log_val(self, agent_id, val):
        pass

    def build_state_offset(self, agent_id):
        state_size = self.get_state_size(agent_id)
        return np.zeros(state_size)

    def build_state_scale(self, agent_id):
        state_size = self.get_state_size(agent_id)
        return np.ones(state_size)

    def build_goal_offset(self, agent_id):
        goal_size = self.get_goal_size(agent_id)
        return np.zeros(goal_size)

    def build_goal_scale(self, agent_id):
        goal_size = self.get_goal_size(agent_id)
        return np.ones(goal_size)

    def build_action_offset(self, agent_id):
        action_size = self.get_action_size()
        return np.zeros(action_size)

    def build_action_scale(self, agent_id):
        action_size = self.get_action_size()
        return np.ones(action_size)

    def build_action_bound_min(self, agent_id):
        action_size = self.get_action_size()
        return -inf * np.ones(action_size)

    def build_action_bound_max(self, agent_id):
        action_size = self.get_action_size()
        return inf * np.ones(action_size)

    def build_state_norm_groups(self, agent_id):
        state_size = self.get_state_size(agent_id)
        return Normalizer.NORM_GROUP_SINGLE * np.ones(state_size, dtype=np.int32)

    def build_goal_norm_groups(self, agent_id):
        goal_size = self.get_goal_size(agent_id)
        return Normalizer.NORM_GROUP_SINGLE * np.ones(goal_size, dtype=np.int32)

    @abstractmethod
    def calc_reward(self, agent_id):
        return 0

    @abstractmethod
    def get_reward_min(self, agent_id):
        return 0

    @abstractmethod
    def get_reward_max(self, agent_id):
        return 1

    @abstractmethod
    def get_reward_fail(self, agent_id):
        return self.get_reward_min(agent_id)

    @abstractmethod
    def get_reward_succ(self, agent_id):
        return self.get_reward_max(agent_id)

    @abstractmethod
    def is_episode_end(self):
        return False

    @abstractmethod
    def check_terminate(self, agent_id):
        return Terminate.Null

    @abstractmethod
    def check_valid_episode(self):
        return True

    @abstractmethod
    def set_sample_count(self, count):
        pass

    @abstractmethod
    def set_mode(self, mode):
        pass

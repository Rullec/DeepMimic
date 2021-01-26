import torch.optim as optim
from learning.torch.replay_buffer_torch import ReplayBufferTorch
from util.logger import Logger
from learning.tf.exp_params import ExpParams
from learning.torch.torch_agent import TorchAgent
from learning.torch.nets.net_builder import build_net
import time
import os
import json
import datetime
import torch
import numpy as np


class TorchIndeAgent(TorchAgent):
    """
        Torch independent-action agent
        each action has its own independent networkd and lr
    """
    NAME = "DiffMBRL_Inde"

    def __init__(self, world, id, json_data):
        super().__init__(world, id, json_data)

    def _build_graph(self, json_data):
        assert self.POLICY_NET_KEY in json_data
        nets = json_data[self.POLICY_NET_KEY]
        assert self.ACTIVATION_KEY in json_data
        activation = json_data[self.ACTIVATION_KEY]
        assert self.DROPOUT_KEY in json_data
        dropout = json_data[self.DROPOUT_KEY]
        self.actions = []

        for i in range(self.get_action_size()):
            net = build_net(nets, self.get_state_size(), 1,
                            activation=activation, dropout=dropout)
            self.actions.append(net)

    def _build_loss(self):
        self.optimizers = []
        for i in range(self.get_action_size()):
            self.optimizers.append(optim.SGD(self.actions[i].parameters(),
                                             lr=self.lr, weight_decay=self.weight_loss))

    def _decide_action(self, s, g):

        actions = []
        for i in range(self.get_action_size()):
            self.actions[i].eval()
            actions.append(self.actions[i](torch.Tensor(s)).detach().numpy())
        actions = np.reshape(np.array(actions), -1)

        assert actions.shape[0] == self.get_action_size(
        ), f"action shape {actions.shape}"
        return actions

    def _train_subloss(self, id):
        assert 0 <= id < self.get_action_size()
        opt = self.optimizers[id]
        opt.zero_grad()
        states = np.array(self.replay_buffer.get_state())
        drdas = np.array(self.replay_buffer.get_drda())[:, id]
        subaction = self.actions[id]
        subaction.train()
        actions_torch = subaction(torch.Tensor(states))
        drdas_torch = torch.Tensor(drdas)
        loss_sum = -torch.mean(drdas_torch * actions_torch)
        loss_sum.backward()

        torch.nn.utils.clip_grad_value_(subaction.parameters(), 3)
        opt.step()

    def load_model(self, path):
        big_dict = torch.load(path)
        for i in range(self.get_action_size()):
            self.actions[i].load_state_dict(big_dict[i])
        print(f"[log] torch load model from {path}")

    def save_model(self, path):
        big_dict = {}
        for _idx, i in enumerate(self.actions):
            big_dict[_idx] = i.state_dict()
        torch.save(big_dict, path)

    def _get_lr(self):
        lrs = []
        for i in self.optimizers:
            for g in i.param_groups:
                lrs.append(g['lr'])
        return lrs

    def _set_lr(self, lr):
        for _idx, opt in enumerate(self.optimizers):
            for g in opt.param_groups:
                if 'lr' in g:
                    g['lr'] = lr[_idx]

    def _train(self):
        for i in range(self.get_action_size()):
            self._train_subloss(i)

        self._set_lr([i * self.lr_decay for i in self._get_lr()])
        self._total_sample_count += self.replay_buffer.get_cur_size()
        self.world.env.set_sample_count(self._total_sample_count)
        self._update_exp_params()

        # 4. output and clear
        output_name = datetime.datetime.now().strftime("%m-%d-%H:%M:%S")
        output_name = f"{output_name}-{str(self.replay_buffer.get_avg_reward())[:6]}.pkl"
        output_path = os.path.join(self.output_dir, output_name)
        self.save_model(output_path)

        cost_time = time.time() - self._begin_time
        avg_rew = self.replay_buffer.get_avg_reward()
        np.set_printoptions(suppress=True)
        print(
            f"[log] total samples {self._total_sample_count} train time {cost_time} s, avg reward {avg_rew}, lr {self._get_lr()[0]}")
        if self._total_sample_count > self.max_samples:
            print(f"[log] total samples exceed max {self.max_samples}, exit")
            exit(0)
        np.set_printoptions(suppress=False)
        self._print_statistics()

        self.replay_buffer.clear()

        self._mode = self.Mode.TRAIN
        self._train_iters += 1

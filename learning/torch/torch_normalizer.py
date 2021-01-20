import torch
import numpy as np


class NormalizerTorch():
    # these group IDs must be the same as those in CharController.h
    NORM_GROUP_SINGLE = 0
    NORM_GROUP_NONE = -1

    def __init__(self, name, size, group_ids=None, alpha=1):
        print(f"{name} group ids {group_ids}")
        self.group_ids = group_ids if group_ids is not None else [
            self.NORM_GROUP_SINGLE for i in range(size)]
        self.alpha = alpha
        self._size = size
        self.name = name
        self.mean = torch.zeros(size)
        self.std = torch.ones(size)
        self.std_min = 0.1
        self.mean.data = torch.Tensor(np.zeros(size)).data
        self.std.data = torch.Tensor(np.ones(size)).data
        self.sample_count = 0

    def normalize(self, x):
        return (x - self.mean) / (self.std)

    def unnormalize(self, x):
        return x * self.std + self.mean

    # def set_mean(self, mean):
    #     assert type(mean) == torch.Tensor
    #     assert mean.shape == self.mean.shape
    #     self.mean = mean

    # def set_std(self, std):
    #     assert type(std) == torch.Tensor
    #     assert std.shape == self.std.shape
    #     assert torch.min(torch.abs(std)) > 1e-2
    #     self.std = std

    # def set_sample_count(self, new_count):
    #     self.sample_count = new_count

    # def get_sample_count(self):
    #     return self.sample_count

    def update(self, mean, std, new_count):
        if type(mean) is np.ndarray or type(std) is np.ndarray:
            mean = torch.Tensor(mean)
            std = torch.Tensor(std)

        # update it
        total_count = self.sample_count + new_count
        # old_weight = self.sample_count / total_count
        # new_weight = new_count / total_count
        old_weight = self.alpha
        new_weight = 1 - old_weight
        for i in range(self._size):
            if self.group_ids[i] == self.NORM_GROUP_SINGLE:
                self.mean[i] = old_weight * self.mean[i] + new_weight * mean[i]
                self.std[i] = old_weight * self.std[i] + new_weight * std[i]
                self.std[i] = torch.clamp(self.std[i], min=self.std_min)
            elif self.group_ids[i] == self.NORM_GROUP_NONE:
                continue
            else:
                assert False, f"unsupported norm group {self.group_ids[i]}"
        self.sample_count = total_count
        # fit the group
        np.set_printoptions(suppress=True)
        print(f"[update] {self.name} mean = {self.mean}")
        print(f"[update] {self.name} std = {self.std}")

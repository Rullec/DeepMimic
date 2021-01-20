import torch
import numpy as np


class NormalizerTorch():
    def __init__(self, name, size):
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

    def set_mean(self, mean):
        assert type(mean) == torch.Tensor
        assert mean.shape == self.mean.shape
        self.mean = mean

    def set_std(self, std):
        assert type(std) == torch.Tensor
        assert std.shape == self.std.shape
        assert torch.min(torch.abs(std)) > 1e-2
        self.std = std

    def set_sample_count(self, new_count):
        self.sample_count = new_count

    def get_sample_count(self):
        return self.sample_count

    def update(self, mean, std):
        if type(mean) is np.ndarray or type(std) is np.ndarray:
            mean = torch.Tensor(mean)
            std = torch.Tensor(std)
        alpha = 0.999
        self.mean = alpha * self.mean + (1 - alpha) * mean
        self.std = alpha * self.std + (1 - alpha) * std
        self.std = torch.clamp(self.std, min=self.std_min)
        np.set_printoptions(suppress=True)
        print(f"[update] {self.name} mean = {self.mean}")
        print(f"[update] {self.name} std = {self.std}")

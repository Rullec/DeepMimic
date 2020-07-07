import numpy as np
from abc import ABCMeta, abstractmethod


class ShapeGen(metaclass=ABCMeta):
    UNIFORM_DISTRIBUTION = 0

    def __init__(self, shape_dim, shape_lb, shape_ub, nn_layers, l2_coeff, proposal, lr_nn, lr_mu, k):
        self.shape_dim = shape_dim
        self.shape_lb = shape_lb
        self.shape_ub = shape_ub
        self.nn_layers = nn_layers
        self.proposal = proposal
        assert len(shape_ub) == shape_dim
        assert len(shape_lb) == shape_dim

        self.l2_coeff = l2_coeff
        self.lr_nn = lr_nn
        self.lr_mu = lr_mu
        self.k = k
        self.mu = 1  # the average value of vsb nn
        # self._build_network()
        # self._build_loss()
        # self._init_tf()
        # self._init_uniform_prob()

    @abstractmethod
    def update(self, sb_input, v_target):
        pass

    def _generate_stage(self):
        sb_prime = np.random.uniform(self.shape_lb, self.shape_ub, self.shape_dim)
        # sb_prime = sb_prime * (self.shape_ub - self.shape_lb) + self.shape_lb
        sb_prime = np.reshape(sb_prime, (1, self.shape_dim))
        return sb_prime

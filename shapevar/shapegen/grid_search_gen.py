import numpy as np

from shapevar.shapegen.shape_gen import ShapeGen


class GridSearchGen(ShapeGen):
    NAME = 'GRID_SEARCH'

    def __init__(self, shape_dim, shape_lb, shape_ub, nn_layers, l2_coeff, proposal, lr_nn, lr_mu, k):
        super().__init__(shape_dim, shape_lb, shape_ub, nn_layers, l2_coeff, proposal, lr_nn, lr_mu, k)
        self.n = 10
        dist = (self.shape_ub - self.shape_lb) / self.n
        self.shape_pool = [dist * i + self.shape_lb for i in range(0, self.n + 1)]
        # return
        # self.shape_pool = []
        # for i in range(self.shape_dim):
        #     for j in range(self.n + 1):
        #         d = self.shape_lb.copy()
        #         d[i] += dist[i] * j
        #         self.shape_pool.append(d)

    def generate_shape(self, sb_input):
        idx = np.random.randint(0, len(self.shape_pool))
        return self.shape_pool[idx]
        # return self.shape_ub

    def update(self, sb_input, v_target):
        return 0

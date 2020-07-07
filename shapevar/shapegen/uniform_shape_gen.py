import numpy as np
from shapevar.shapegen.shape_gen import ShapeGen


class UniformShapeGen(ShapeGen):
    NAME = 'UNIFORM_GEN'

    def __init__(self, shape_dim, shape_lb, shape_ub, nn_layers, l2_coeff, proposal, lr_nn, lr_mu, k):
        super().__init__(shape_dim, shape_lb, shape_ub, nn_layers, l2_coeff, proposal, lr_nn, lr_mu, k)

    def generate_shape(self, sb_input):
        return self._generate_stage()

    def update(self, sb_input, v_target):
        return 0


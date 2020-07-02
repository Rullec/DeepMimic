from learning.ppo_agent import PPOAgent
from shapevar.shapegen.shape_gen import ShapeGen
import numpy as np


class PPOShapeVarAgent(PPOAgent):
    NAME = 'PPO_SHAPE_VAR'
    SHAPE_VAR_AGENT_KEY = 'ShapeVarAgent'

    def __init__(self, world, id, json_data):
        super().__init__(world, id, json_data)
        self.state_size = self.world.env.get_state_size(id)

        self.var_links = self.world.env.get_var_links()
        self.var_links_names_id_map = self.world.env.get_var_links_names_id_map()

        self._compute_unique_variable_size()
        self._compute_shape_bounds()
        self._init_state_of_body_shape()
        self._init_unique_variable_link_state_idx()

        assert PPOShapeVarAgent.SHAPE_VAR_AGENT_KEY in json_data
        shape_var_agent_data = json_data[PPOShapeVarAgent.SHAPE_VAR_AGENT_KEY]

        layers = shape_var_agent_data['layers']
        l2_coeff = shape_var_agent_data['l2_coeff']
        lr_nn = shape_var_agent_data['lr_nn']
        lr_mu = shape_var_agent_data['lr_mu']
        k = shape_var_agent_data['k']

        # layers = [512, 256, 64]
        # l2_coeff = 1
        # lr_nn = 1e-3
        # lr_mu = 1e-2
        # k = 1
        proposal_func = ShapeGen.UNIFORM_DISTRIBUTION
        self.shape_generator = ShapeGen(self.n_unique_var_shape_size, self.lb, self.ub, layers, l2_coeff, proposal_func,
                                        lr_nn, lr_mu, k)

        self.generate_new_body_shape()

    def reshape_state(self, s):
        s = np.hstack([s, self.sb])
        return np.reshape(s, [-1, self.state_size])

    def concat_state(self, s):
        return np.hstack([s, self._recover_body_shape_param(self.sb)])

    def _record_state(self):
        s = self.world.env.record_state(self.id)
        return self.concat_state(s)

    def generate_new_body_shape(self):
        sb_prime = self.shape_generator.generate_shape(sb_input=self.sb)
        self.sb = sb_prime
        if self.world.env.is_symmetric_var_mode():
            sb_prime = self._recover_body_shape_param(sb_prime)
        return sb_prime

    def _compute_unique_variable_size(self):
        """"
            compute the real unique variable for sampling
            if sampling in symmetric mode
            this func will return the number of symmetric root
            e.g: if set symmetric sampling, and we set the 'SymLink' attribute of LeftLeg 'RightLeg' in the config file
            the body shape of LeftLeg will be same as RightLeg, and we will ignore the LeftLeg in sampling.
        """
        self.n_unique_var_links = 0
        self.n_unique_var_shape_size = 0
        self.unique_links = dict()
        self.body_shape_dim = 0
        cnt = 0
        for link in self.var_links:
            name = link.sym_link if self.world.env.is_symmetric_var_mode() else link.name
            if not self.unique_links.__contains__(name):
                self.unique_links[link.name] = cnt
                cnt += 1
            # this is the real size of body param which is sent to controller nn
            self.body_shape_dim += 3

        self.n_unique_var_links = len(self.unique_links)
        # this is the size of body param which is sent to marginal value function nn
        self.n_unique_var_shape_size = self.n_unique_var_links * 3
        print(self.unique_links)
        print('n_unique_var_links     : {}'.format(self.n_unique_var_links))
        print('n_unique_var_shape_size: {}'.format(self.n_unique_var_shape_size))

    def _compute_shape_bounds(self):
        lb = []
        ub = []
        for link_name in self.unique_links:
            link = self.var_links[self.var_links_names_id_map[link_name]]
            lb.append(link.lb)
            ub.append(link.ub)
        self.lb = np.array(np.hstack(lb))
        self.ub = np.array(np.hstack(ub))
        print('lb: {}'.format(self.lb))
        print('ub: {}'.format(self.ub))

    def _recover_body_shape_param(self, sb):
        sb = np.squeeze(sb)
        sb_prime = []
        for link in self.var_links:
            for i in range(self.unique_links[link.sym_link] * 3, (self.unique_links[link.sym_link] + 1) * 3):
                sb_prime.append(sb[i])
        return sb_prime

    def _init_state_of_body_shape(self):
        self.sb = [1 for _ in range(self.n_unique_var_shape_size)]

    def _init_unique_variable_link_state_idx(self):
        self.unique_variable_link_state_idx = []
        for link_name in self.unique_links.keys():
            self.unique_variable_link_state_idx.append(self.state_size - self.body_shape_dim + self.var_links_names_id_map[link_name] * 3)
            self.unique_variable_link_state_idx.append(self.state_size - self.body_shape_dim + self.var_links_names_id_map[link_name] * 3 + 1)
            self.unique_variable_link_state_idx.append(self.state_size - self.body_shape_dim + self.var_links_names_id_map[link_name] * 3 + 2)
        print('unique_variable_link_state_idx: {}'.format(self.unique_variable_link_state_idx))

    def _update_generator(self, s, g, tar_vals):
        # step 1. Gather the state of unique variable links
        s_ = s[:, self.unique_variable_link_state_idx]
        # step 2. Update generator
        tar_vals = np.reshape(tar_vals, (-1, 1))
        return self.shape_generator.update(s_, v_target=tar_vals)
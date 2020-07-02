import json
import numpy as np
from env.deepmimic_env import DeepMimicEnv


class DeepMimicVariableLink:
    name = ""
    ub = []
    lb = []
    id = 0
    sym_link = ""

    def __init__(self, id, name, lb, ub, sym_link, is_fixed=False):
        self.id         = id
        self.name       = name
        self.lb         = lb
        self.ub         = ub
        self.sym_link   = sym_link
        self.is_fixed   = is_fixed

    def print(self):
        print('id     : {}'.format(self.id))
        print('name   : {}'.format(self.name))
        print('symlink: {}'.format(self.sym_link))
        print('lb     : {}'.format(self.lb))
        print('ub     : {}'.format(self.ub))
        print('is_fixed:{}'.format(self.is_fixed))


class DeepMimicShapeVarEnv(DeepMimicEnv):
    _is_symmetric_var = False
    var_links_name_id_map = dict()
    var_links = []
    body_shape_dim = 0

    def __init__(self, args, enable_draw):
        super().__init__(args, enable_draw)
        self._load_var_link_json()

    def _load_var_link_json(self):
        path = self._core.GetStringArgs("var_links_files")
        self._is_symmetric_var = self._core.GetBooleanArgs("is_symmetric_var")
        self.var_links_data = json.load(open(file=path, mode='r'))

        for _id, link in enumerate(self.var_links_data['var_links']):
            self.var_links_name_id_map[link] = _id

        for link in self.var_links_data['var_links']:
            link_attrib = self.var_links_data['Attribute'][link]
            lb = link_attrib['lb']
            ub = link_attrib['ub']
            sym_link = link_attrib['SymLink']
            is_fixed = link_attrib['is_fixed']
            _id = self.var_links_name_id_map[link]
            var_link = DeepMimicVariableLink(_id, link, lb, ub, sym_link, is_fixed)
            self.var_links.append(var_link)
            self.body_shape_dim += 3

    def get_state_size(self, agent_id):
        return self._core.GetStateSize(agent_id) + self.body_shape_dim

    def build_state_offset(self, agent_id):
        offset = self._core.BuildStateOffset(agent_id)
        return self._append_zero(offset)

    def build_state_scale(self, agent_id):
        scale = self._core.BuildStateScale(agent_id)
        return self._append_one(scale)

    def _append_one(self, v):
        v = list(v)
        for _ in range(self.body_shape_dim):
            v.append(1)
        return np.array(v)

    def _append_zero(self, v):
        v = list(v)
        for _ in range(self.body_shape_dim):
            v.append(0)
        return np.array(v)

    def init_state_of_body_shape(self):
        return np.array([1.0 for _ in range(self.body_shape_dim)])

    def build_state_norm_groups(self, agent_id):
        '''todo: fix here'''
        group = self._core.BuildStateNormGroups(agent_id)
        group = list(group)
        for _ in range(self.body_shape_dim):
            group.append(0)
        return tuple(group)

    def is_symmetric_var_mode(self):
        return self._is_symmetric_var

    def get_var_links(self):
        return self.var_links

    def get_var_links_names_id_map(self):
        return self.var_links_name_id_map

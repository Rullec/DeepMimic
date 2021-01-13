import tensorflow as tf
import pickle as pl
import numpy as np
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import os

good_model = "/home/xudong/Projects/SupervisedRL/output/legs_paths_mimic/model/mimic_pg_net.ckpt-150"

exp_model = "/home/xudong/Projects/DeepMimic/output/1009/exp/agent0_model_2020_10_12_14_13_01_11.40.ckpt"
# pickle_model = "/home/xudong/Projects/DeepMimic/output/1009/exp/agent0_model_2020_10_12_14_13_01_11.40.ckpt.weight"


# with open(pickle_model, "rb") as f:
#     pickel_variable = pl.load(f)

good_vars_name_shape = tf.train.list_variables(good_model)
exp_vars_name_shape = tf.train.list_variables(exp_model)

all_vars = tf.global_variables()


def find(name, name_shape_list):
    for i in name_shape_list:
        if name == i[0]:
            return i[1]
    return None


print_tensors_in_checkpoint_file(
    file_name=good_model, tensor_name="agent/resource/a_norm/count", all_tensors=False)
print_tensors_in_checkpoint_file(
    file_name=good_model, tensor_name="agent/resource/a_norm/mean", all_tensors=False)

for var in good_vars_name_shape:
    name = var[0]
    shape = var[1]

    exp_shape = find(name, exp_vars_name_shape)

    if exp_shape is not None:
        print(
            f"[tf] good model var {name}, shape {shape} in exp shape {exp_shape}")
        assert shape == exp_shape
    else:
        print(f"fail to find {name}")
        exit()


# for i in pickel_variable:
#     value = pickel_variable[i]
#     print(f"[pkl] load var {i}, shape {np.shape(value)}")
#     continue

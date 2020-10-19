import numpy as np
import tensorflow as tf
from abc import abstractmethod

from learning.rl_agent import RLAgent
from util.logger import Logger
from learning.tf_normalizer import TFNormalizer
import datetime
import pickle


class TFAgent(RLAgent):
    """
        TFAgent 又是RLAgent的子类

    """

    RESOURCE_SCOPE = "resource"
    SOLVER_SCOPE = "solvers"

    def __init__(self, world, id, json_data):
        self.tf_scope = "agent"
        # TFAgent是RLAgent的子类
        self.graph = tf.Graph()  # 定义了graph，然后在后面可供调用?
        self.sess = tf.Session(graph=self.graph)

        # json_data: agent file 中读进来的json，不是骨架结构
        super().__init__(world, id, json_data)
        self._build_graph(json_data)
        self._init_normalizers()
        return

    def __del__(self):
        self.sess.close()
        return

    def save_model(self, out_path):
        # save model
        with self.sess.as_default(), self.graph.as_default():
            try:
                save_path = self.saver.save(
                    self.sess, out_path, write_meta_graph=False, write_state=False
                )
            except:
                Logger.print("Failed to save model to: " + save_path)

        # save weight
        weight_lst = self._tf_vars("main/actor/")
        weight_dict = {}
        name_lst = []
        size = 0
        for i in weight_lst:
            name_lst.append(i.name)
            weight_dict[i.name] = self.sess.run(i)
            # print((i.name, weight_dict[i.name].shape))
            size += weight_dict[i.name].size
        # print("sum size = %d" % size)
        weight_save_path = save_path + ".weight"
        with open(weight_save_path, "wb") as f:
            pickle.dump(weight_dict, f)

        Logger.print("Model saved to: " + save_path)
        Logger.print("Model weight saved to : " + weight_save_path)
        return

    def load_model(self, in_path):
        with self.sess.as_default(), self.graph.as_default():
            self.saver.restore(self.sess, in_path)
            self._load_normalizers()
            Logger.print("Model loaded from: " + in_path)
        return

    def _get_output_path(self):
        assert self.output_dir != ""
        self.train_return
        file_path = (
            self.output_dir
            + "/agent"
            + str(self.id)
            + "_model_"
            + str(datetime.datetime.now())[:19]
            .replace(" ", "_")
            .replace("-", "_")
            .replace(":", "_")
            + str("_%.2f" % self.avg_train_return)
            + ".ckpt"
        )
        return file_path

    def _get_int_output_path(self):
        assert self.int_output_dir != ""
        file_path = self.int_output_dir + (
            "/agent{:d}_models/agent{:d}_int_model_{:010d}.ckpt"
        ).format(self.id, self.id, self.iter)
        return file_path

    def _build_graph(self, json_data):
        with self.sess.as_default(), self.graph.as_default():
            with tf.variable_scope(self.tf_scope):
                self._build_nets(json_data)

                with tf.variable_scope(self.SOLVER_SCOPE):
                    self._build_losses(json_data)
                    self._build_solvers(json_data)

                self._initialize_vars()
                self._build_saver()

        return

    def _init_normalizers(self):
        with self.sess.as_default(), self.graph.as_default():
            # update normalizers to sync the tensorflow tensors
            self.s_norm.update()
            self.g_norm.update()
            self.a_norm.update()
        return

    @abstractmethod
    def _build_nets(self, json_data):
        pass

    @abstractmethod
    def _build_losses(self, json_data):
        pass

    @abstractmethod
    def _build_solvers(self, json_data):
        pass

    def _tf_vars(self, scope=""):
        # 获得某个scope中的所有变量
        with self.sess.as_default(), self.graph.as_default():
            res = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.tf_scope + "/" + scope
            )
            assert len(res) > 0
        return res

    def _build_normalizers(self):
        with self.sess.as_default(), self.graph.as_default(), tf.variable_scope(
            self.tf_scope
        ):
            with tf.variable_scope(self.RESOURCE_SCOPE):
                self.s_norm = TFNormalizer(
                    self.sess,
                    "s_norm",
                    self.get_state_size(),
                    self.world.env.build_state_norm_groups(self.id),
                )
                self.s_norm.set_mean_std(
                    -self.world.env.build_state_offset(self.id),
                    1 / self.world.env.build_state_scale(self.id),
                )
                self.g_norm = TFNormalizer(
                    self.sess,
                    "g_norm",
                    self.get_goal_size(),
                    self.world.env.build_goal_norm_groups(self.id),
                )
                self.g_norm.set_mean_std(
                    -self.world.env.build_goal_offset(self.id),
                    1 / self.world.env.build_goal_scale(self.id),
                )

                # 初始化action normalizer
                self.a_norm = TFNormalizer(
                    self.sess, "a_norm", self.get_action_size())

                # set mean and std for actions
                # mean = - offset, mean that action is around [mean], but offset means that action + offset = 0
                # std = 1.0 / scale. std means that action is in [-std, std], but scale means action * scale = [-1, 1]

                # so, mean = - offset
                # std = 1.0 / scale
                self.a_norm.set_mean_std(
                    -self.world.env.build_action_offset(self.id),
                    1 / self.world.env.build_action_scale(self.id),
                )
        return

    def _load_normalizers(self):
        self.s_norm.load()
        self.g_norm.load()
        self.a_norm.load()
        return

    def _update_normalizers(self):
        with self.sess.as_default(), self.graph.as_default():
            super()._update_normalizers()
        return

    def _initialize_vars(self):
        self.sess.run(tf.global_variables_initializer())
        return

    def _build_saver(self):
        vars = self._get_saver_vars()
        # [print(i) for i in vars]
        self.saver = tf.train.Saver(vars, max_to_keep=0)
        return

    def _get_saver_vars(self):
        with self.sess.as_default(), self.graph.as_default():
            vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=self.tf_scope)
            vars = [v for v in vars if "/" +
                    self.SOLVER_SCOPE + "/" not in v.name]
            assert len(vars) > 0
        return vars

    def _weight_decay_loss(self, scope):
        vars = self._tf_vars(scope)
        vars_no_bias = [v for v in vars if "bias" not in v.name]
        loss = tf.add_n(
            [tf.nn.l2_loss(v) for v in vars_no_bias]
        )  # 对于这些变量，全部做成Loss，也就是让weight越小越好。
        return loss

    def _train(self):
        with self.sess.as_default(), self.graph.as_default():
            super()._train()
        return

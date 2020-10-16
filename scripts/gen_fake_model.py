import learning.nets.net_builder as NetBuilder
from learning.tf_normalizer import TFNormalizer
import tensorflow as tf
import sys
import numpy as np
sys.path.append("..")


class DIYAgent:
    def __init__(self):
        # 1. create the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        # 2. build net
        self._load_params()
        self._build_graph()
        return

    def _load_params(self):
        self.FC_NAME = "fc_2layers_512_256"
        self.RESOURCE_SCOPE = "resource"
        self.STATE_SIZE = 93
        self.ACTION_SIZE = 18
        self.GOAL_SIZE = 0
        self.VAL_SIZE = 1

        # 1. build group
        self.state_group_id = np.ones(self.STATE_SIZE, dtype=np.int32)
        self.state_group_id[0] = 0
        self.action_group_id = np.ones(self.ACTION_SIZE, dtype=np.int32)
        self.val_group_id = np.ones(self.VAL_SIZE, dtype=np.int32)
        self.goal_group_id = np.ones(self.GOAL_SIZE, dtype=np.int32)

        # 2. constants
        self.tf_scope = "agent"

    def _build_graph(self):
        with self.sess.as_default(), self.graph.as_default():
            with tf.variable_scope(self.tf_scope):
                self._build_normalizer()
                self._build_nets()

                # initialize the graph
                self._initialize_vars()
                # build the saver
                self._build_saver()

    def _build_saver(self):
        vars = self._get_saver_vars()
        # [print(i) for i in vars]
        self.saver = tf.train.Saver(vars, max_to_keep=0)
        return

    def _get_saver_vars(self):
        with self.sess.as_default(), self.graph.as_default():
            vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=self.tf_scope)
            assert len(vars) > 0
        return vars

    def _build_nets(self):
        # input
        self.s_tf = tf.placeholder(
            tf.float32, shape=[None, self.STATE_SIZE], name="s"
        )  # 输入state
        self.a_tf = tf.placeholder(
            tf.float32, shape=[None, self.ACTION_SIZE], name="a"
        )  # 输入action

        self.g_tf = tf.placeholder(tf.float32, shape=(None), name="g")  # goal

        with tf.variable_scope("main"):
            with tf.variable_scope("actor"):
                self.a_mean_tf = self._build_net_actor(self.FC_NAME, 1.0)
            with tf.variable_scope("critic"):
                self.critic_tf = self._build_net_ciritc(self.FC_NAME, 1.0)
        return

    def _initialize_vars(self):
        self.sess.run(tf.global_variables_initializer())
        return

    def _build_net_actor(self, fc_name, init_output_scale):
        """
            build the actor network, input state, output action.
            state -> normalized state -> FC -> normalized action -> action

            the goal of normalization is to make the training & learning easier

        """
        # 1. call the state normalizer, normalize the input state
        norm_s_tf = self.s_norm.normalize_tf(self.s_tf)
        input_tfs = [norm_s_tf]

        # 2. build the FCs
        h = NetBuilder.build_net(fc_name, input_tfs)

        # 3. concanate the output normed action
        norm_a_tf = tf.layers.dense(
            inputs=h,
            units=self.ACTION_SIZE,
            activation=None,
            kernel_initializer=tf.random_uniform_initializer(
                minval=-init_output_scale, maxval=init_output_scale
            ),
        )

        # 4. unnormalize the action then return
        a_tf = self.a_norm.unnormalize_tf(norm_a_tf)

        return a_tf

    def _build_net_ciritc(self, fc_name, init_output_scale):
        # 1. call the state normalizer, normalize the input state
        norm_s_tf = self.s_norm.normalize_tf(self.s_tf)
        input_tfs = [norm_s_tf]

        # 2. build the FCs
        h = NetBuilder.build_net(fc_name, input_tfs)

        # 3. concanate the output normed val
        norm_val_tf = tf.layers.dense(
            inputs=h,
            units=self.VAL_SIZE,
            activation=None,
            kernel_initializer=tf.random_uniform_initializer(
                minval=-init_output_scale, maxval=init_output_scale
            ),
        )

        # 4. unnormalize the action then return
        norm_val_tf = tf.reshape(norm_val_tf, [-1])
        val_tf = self.val_norm.unnormalize_tf(norm_val_tf)
        return val_tf

    def _build_normalizer(self):
        """
        Build the normalizer for state, action, value, goal
        """
        with self.sess.as_default(), self.graph.as_default():
            with tf.variable_scope(self.RESOURCE_SCOPE):
                self.s_norm = TFNormalizer(
                    self.sess, "s_norm", self.STATE_SIZE, self.state_group_id
                )
                self.s_norm.set_mean_std(
                    np.zeros(self.STATE_SIZE), np.ones(self.STATE_SIZE)
                )

                self.a_norm = TFNormalizer(
                    self.sess, "a_norm", self.ACTION_SIZE, self.action_group_id
                )
                self.a_norm.set_mean_std(
                    np.zeros(self.ACTION_SIZE), np.ones(self.ACTION_SIZE)
                )

                self.val_norm = TFNormalizer(
                    self.sess, "val_norm", self.VAL_SIZE, self.val_group_id
                )
                self.val_norm.set_mean_std(
                    np.zeros(self.VAL_SIZE), np.ones(self.VAL_SIZE)
                )

                self.g_norm = TFNormalizer(
                    self.sess, "g_norm", self.GOAL_SIZE, self.goal_group_id
                )
                self.g_norm.set_mean_std(
                    np.zeros(self.GOAL_SIZE), np.ones(self.GOAL_SIZE)
                )

        return

    def get_state_size(self):
        return self.STATE_SIZE

    def save_model(self, path):
        print(f"save model to {path}")
        self.saver.save(self.sess, path)
        return


agent = DIYAgent()
agent.save_model("hello")
# with tf.variable_scope("main"):
# with tf.variable_scope("actor"):

# 1. given model file, build the actor network

# 2. build the critic net

# 3. build normalizer

# 4. output the weight

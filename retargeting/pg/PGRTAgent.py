from learning.tf.pg_agent import PGAgent


class PGRTAgent(PGAgent):
    NAME = 'PGRT'

    def __init__(self, world, id, json_data):
        super().__init__(world, id, json_data)

    def run_train(self):
        pass
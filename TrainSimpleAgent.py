from DeepMimic import build_arg_parser
from learning.agent_builder import build_agent
import sys


def build(args):
    arg_parser = build_arg_parser(args)
    agent_file = arg_parser.parse_string('agent_files')
    agent = build_agent(None, 0, agent_file)
    return agent


def main():
    global args

    # Command line arguments
    args = sys.argv[1:]
    agent = build(args)
    agent.test()
    return

if __name__ == '__main__':
    main()
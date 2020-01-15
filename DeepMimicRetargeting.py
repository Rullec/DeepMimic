import sys
from DeepMimic import reload, build_world


def main():
    global args
    args = sys.argv[1:]

    world = build_world(args, enable_draw=False)


if __name__ == '__main__':
    main()
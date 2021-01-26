import matplotlib.pyplot as plt
import re
import numpy as np
import argparse
from scripts.vis_torch_log import get_all_log_files
import os

def parse_np_prefix(prefix, cont):
    pat = re.compile(rf"(?<={prefix})[\.\s\d]*(?=\])", re.M | re.S)

    res_lst = []
    for l in pat.findall(cont):
        res_lst.append([float(i) for i in l.split()])
    return res_lst


def handle_log_file(filename, output_name, draw=False):
    print(f"handle {filename}")
    with open(filename) as f:
        cont = f.read()
    plt.cla()
    plt.clf()
    state_std = np.array(parse_np_prefix(r"state std \[", cont))
    action_std = np.array(parse_np_prefix(r"action std \[", cont))
    drda_std = np.array(parse_np_prefix(r"drda std \[", cont))
    state_std_norm = np.linalg.norm(state_std, axis=1)
    action_std_norm = np.linalg.norm(action_std, axis=1)
    drda_std_norm = np.linalg.norm(drda_std, axis=1)

    plt.subplot(1, 3, 1)
    plt.plot(state_std_norm)
    plt.title("state")
    plt.subplot(1, 3, 2)
    plt.plot(action_std_norm)
    plt.title("action")
    plt.subplot(1, 3, 3)
    plt.plot(drda_std_norm)
    plt.title("drda")

    plt.suptitle(f"{filename} std")
    if draw:
        plt.show()
    else:
        plt.savefig(output_name)
        print(f"output {output_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vis_log_parser")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    arg = parser.parse_args()

    arg = parser.parse_args()
    if arg.log_file is not None:
        # run in single file mode
        handle_log_file(arg.log_file, arg.log_file, draw=True)
    else:
        log_dir = arg.log_dir
        output_dir = arg.output_dir
        assert log_dir is not None and os.path.exists(log_dir)
        assert output_dir is not None and (os.path.exists(output_dir) is False)

        os.makedirs(output_dir)
        files = get_all_log_files(log_dir)
        for file in files:
            output = os.path.join(output_dir, f"{os.path.split(file)[-1]}.png")
            # print(output)
            handle_log_file(file, output, draw=False)

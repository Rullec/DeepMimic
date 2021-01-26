import matplotlib.pyplot as plt
import sys
import os
import argparse


def handle_log_file(log_filename, output_png_filename, draw=False):
    assert os.path.exists(log_filename) == True
    with open(log_filename) as f:
        cont = f.readlines()

    samples_lst = []
    avg_rew_lst = []
    lr_lst = []
    time_lst = []
    for line in cont:
        if line.find("total samples") != -1:
            splited = line.split()
            # samples_lst.append (int())
            try:
                samples = int(splited[3])
                time = splited[6]
                r = float(splited[10][:-1])
                lr = float(splited[12])
                # print(f"sample {samples}, t {time}, r {r} lr {lr}")
                samples_lst.append(samples)
                avg_rew_lst.append(r)
                lr_lst.append(lr)
            except Exception as e:
                print(f"{e}, continue")
                continue
        if line.find("timer") != -1:
            splited = line.split()

            try:
                time = float(splited[10][:-1])
                # print(time)
            except Exception as e:
                print(f"{e}, continue")
                continue
            time_lst.append(time)
    plt.clf()
    plt.suptitle(output_png_filename)
    plt.subplot(1, 3, 1)
    # plt.ylim(0, 1)
    # plt.plot(samples_lst, avg_rew_lst)
    plt.plot([i for i in range(len(avg_rew_lst))], avg_rew_lst)
    plt.title(f"reward")
    plt.subplot(1, 3, 2)
    # plt.plot(samples_lst, lr_lst)
    plt.plot([i for i in range(len(lr_lst))], lr_lst)
    plt.title(f"lr")
    plt.subplot(1, 3, 3)
    plt.plot( time_lst)
    plt.title(f"timer")
    if draw is True:
        plt.show()
    else:
        plt.savefig(output_png_filename)
    print(f"[log] {output_png_filename} succ")


def get_all_log_files(logdir):
    assert logdir is not None
    lst = []
    for i in os.listdir(logdir):
        print(i)
        if i.find(".log") != -1:
            lst.append(os.path.join(logdir, i))

    if len(lst) == 0:
        return None
    else:
        return lst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vis_log_parser")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

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
        # file = "../logs/train_diff_legs_stdagent_diffstd_lr_0.0001_lr_decay_0.97_replay_size_100.json.txt.log"

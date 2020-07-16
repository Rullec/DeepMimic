import matplotlib.pyplot as plt
import sys


def read_log_file(filename):
    try:
        cont = None
        with open(filename, "r") as f:
            cont = f.readlines()

        timer = []
        time_count = []
        for line in cont:
            if line.find("Train_Ret") != -1:
                if len(timer) != 0:
                    time_count.append(float(timer[-1]))
                    timer.clear()

            if line.find("Timer") != -1:
                timer.append(line.split()[8][:-1])
        t_last = 0
        for i, t in enumerate(time_count):
            if abs(t - t_last) > 1:
                print('timer sudden change at {} itr, from {} to {} '.format(i, t_last, t))
                break
            t_last = t

        return time_count

    except FileExistsError:
        print("file %s is not exist" % filename)
        return None


if __name__ =="__main__":
    # print("succ")
    args = sys.argv[1:] # file list

    # check file valid
    for filename in args:
        timer = read_log_file(filename)
        plt.subplot(1, 1, 1)
        plt.plot(timer, label=filename + " timer")
        plt.legend()
        # plt.legend([filename + " train_ret", filename + " test_ret", filename + " timer", ])
    # plt.legend(args)
    plt.show()

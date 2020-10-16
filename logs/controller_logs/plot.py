import matplotlib.pyplot as plt
import sys
import subprocess
# with open("plot.txt") as f:
#     cont = f.readlines()
#     num = []
#     for i in cont:
#         num.append(float(i))
#     print(num)
#     plt.plot(num)
#     plt.show()


def read_log_file(filename):
    try:
        cont = None
        with open(filename, "r") as f:
            cont = f.readlines()

        train_return = []
        test_return = []
        time_count = []
        time_exp_count = []
        time_buffer = []
        time_exp_buffer = []
        for line in cont:
            if line.find("Train_Ret") != -1:
                try:
                    train_return.append(float(line.split()[3]))
                except ValueError:
                    print("[log] line err = " + line)
                    continue
                if len(time_buffer) != 0:
                    time_count.append(float(time_buffer[-1]))
                    time_buffer.clear()
                    time_exp_count.append(float(time_exp_buffer[-1]))
                    time_exp_buffer.clear()
            if line.find("Test_Ret") != -1:
                test_return.append(float(line.split()[3]))
            if line.find("Timer") != -1:  # get timer
                time_buffer.append(line.split()[8][:-1])
                time_exp_buffer.append(line.split()[11])
                # print(line.split())
                # print(time)
        # print((time_count))
        # print(time_exp_count)
        # print((train_return))
        # print((test_return))

        return train_return, test_return, time_count, time_exp_count
    #     cmd = "cat %s | grep -i train_return | awk '{print $4}' | grep -v =" % filename
    #     ret = subprocess.getoutput(cmd).split()
    #     ret = [float(i) for i in ret]

    #     cmd = "cat %s | grep -i timer | awk '{print $7}' | grep -v [a-z]" % filename
    #     time_series = subprocess.getoutput(cmd).split()
    #     gap_divide = int(len(time_series) / len(ret))
    #     times = []
    #     for i, cont in enumerate(time_series):
    #         if i % gap_divide ==0:
    #             print(cont)
    #             times.append(float(cont))

    #     return ret, times

    except FileExistsError:
        print("file %s is not exist" % filename)
        return None


if __name__ == "__main__":
    # print("succ")
    args = sys.argv[1:]  # file list

    # check file valid
    for filename in args:
        train, test, time, time_exp = read_log_file(filename)
        plt.subplot(1, 2, 1)
        plt.plot(train, label=filename + " train_ret")
        # plt.plot(test, label = filename + " test_ret")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(time, label=filename + " timer")
        plt.plot(time_exp, label=filename + " timer exp")
        plt.legend()
        # plt.legend([filename + " train_ret", filename + " test_ret", filename + " timer", ])
    # plt.legend(args)
    plt.show()

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
        clip_frac_buffer = []
        actor_loss = []
        critic_loss = []
        lr = []
        sample_count = []

        for line in cont:
            try:
                if line.find("Train_Ret") != -1:
                    try:
                        train_return.append(float(line.split()[3]))
                    except ValueError:
                        print("[log] line err = " + line)
                        continue
                    if len(time_buffer) != 0:
                        time_count.append(float(time_buffer[-1]))
                        time_buffer.clear()
                        time_exp_count.append(float(time_exp_buffer[-1][:-1]))
                        time_exp_buffer.clear()
                if line.find("Test_Ret") != -1:
                    test_return.append(float(line.split()[3]))
                if line.find("Timer said") != -1:  # get timer
                    time_buffer.append(line.split()[8][:-1])
                    time_exp_buffer.append(line.split()[11])
                if line.find("Clip") != -1:  # get timer
                    clip_frac_buffer.append(float(line.split()[3]))
                if line.find("Actor_Loss") != -1:  # get timer
                    actor_loss.append(float(line.split()[3]))
                if line.find("Critic_Loss") != -1:  # get timer
                    critic_loss.append(float(line.split()[3]))
                if line.find("Samples |") != -1:  # sample count
                    sample_count.append(float(line.split()[3])/1e4)
                if line.find("Actor_Stepsize") != -1:  # learning rate (actor)
                    lr.append(float(line.split()[3]))
            except:
                continue
                # print(line.split())
                # print(time)
        # print((time_count))
        # print(time_exp_count)
        # print((train_return))
        # print((test_return))

        return train_return, test_return, time_count, time_exp_count, clip_frac_buffer, actor_loss, critic_loss, sample_count, lr
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


def cal_avg_ret(ret_lst, time_lst):
    length = min(len(ret_lst), len(time_lst))
    avg_lst = []
    for id in range(length):
        avg_lst.append(ret_lst[id] / time_lst[id])
    return avg_lst


if __name__ == "__main__":
    # print("succ")
    args = sys.argv[1:]  # file list

    # check file valid
    for filename in args:
        train, test, time, time_exp, clip_frac, actor_loss, critic_loss, sample_count, learning_rate = read_log_file(
            filename)
        plt.subplot(3, 3, 1)
        plt.plot(train, label="train")
        plt.plot(test, label="test")
        plt.title("return")
        plt.legend()
        plt.subplot(3, 3, 2)
        plt.plot(time, label="time")
        plt.plot(time_exp, label="exp")
        plt.title("time")
        plt.legend()
        avg_train_ret_lst = cal_avg_ret(train, time)
        plt.subplot(3, 3, 3)
        plt.plot(avg_train_ret_lst)
        plt.title("avg train ret")
        plt.subplot(3, 3, 4)
        plt.plot(clip_frac)
        plt.title("clip fraction")
        plt.subplot(3, 3, 5)
        plt.plot(actor_loss)
        plt.title("actor_loss")
        plt.subplot(3, 3, 6)
        plt.plot(critic_loss)
        plt.title("critic_loss")

        plt.subplot(3, 3, 7)
        plt.plot(learning_rate)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.title("actor_lr")

        plt.subplot(3, 3, 8)
        plt.plot(sample_count)
        plt.title("sample_counts")

        plt.suptitle(filename)

    plt.tight_layout()
    plt.show()

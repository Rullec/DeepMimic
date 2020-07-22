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
        critic_loss = []
        actor_loss = []
        clip_frac = []
        gen_loss = []
        actor_lr = []
        for line in cont:
            if line.find("Train_Ret") != -1 and line.find("Per_Sec") != -1:
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
            if line.find('Critic_Loss') != -1:
                critic_loss.append(float(line.split()[3]))
            if line.find('Actor_Loss') != -1:
                actor_loss.append(float(line.split()[3]))
            if line.find('Clip_Frac') != -1:
                clip_frac.append(float(line.split()[3]))
            if line.find('Generator_Loss') != -1:
                gen_loss.append(float(line.split()[3]))
            if line.find('Actor_Stepsize') != -1:
                actor_lr.append(float(line.split()[3]))
                # print(line.split())
                # print(time)
        # print(time_count)
        # print(time_exp_count)
        # print(train_return)
        # print(test_return)
        t_last = time_count[0]
        for i, t in enumerate(time_count):
            if abs(t - t_last) > 1:
                print('timer sudden change at {} itr, from {} to {} '.format(i, t_last, t))
            t_last = t

        tr_last = train_return[0]
        for i, tr in enumerate(train_return):
            if abs(tr - tr_last) > 20 and tr < tr_last:
                print('train return sudden change at {} itr, from {} to {} '.format(i, tr_last, tr))
                break
            tr_last = tr

        for i, a_loss in enumerate(actor_loss):
            if a_loss > 0.4:
                print('actor loss exploded {} itr, the value is {}'.format(i, a_loss))

        return train_return, test_return, time_count, time_exp_count, critic_loss, actor_loss, clip_frac, gen_loss, actor_lr
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
        train, test, time, time_exp, c_loss, a_loss, clip_frac, gen_loss, actor_lr = read_log_file(filename)
        plt.subplot(2, 3, 1)
        plt.plot(train, label=filename + " train_ret")
        plt.plot(test, label=filename + " test_ret")
        plt.legend()
        plt.subplot(2, 3, 2)
        plt.plot(time, label=filename + " timer")
        plt.plot(time_exp, label=filename + " timer exp")
        plt.legend()
        plt.subplot(2, 3, 4)
        plt.plot(c_loss, label=filename + ' c loss')
        plt.plot(a_loss, label=filename + ' a loss')
        plt.legend()
        plt.subplot(2, 3, 3)
        plt.plot(clip_frac, label=filename + ' clip_frac')
        plt.legend()
        plt.subplot(2, 3, 5)
        plt.plot(gen_loss, label=filename+' gen loss')
        plt.legend()
        plt.subplot(2, 3, 6)
        plt.plot(actor_lr, label=filename+' actor lr')
        plt.legend()
    # plt.legend([filename + " train_ret", filename + " test_ret", filename + " timer", ])
    # plt.legend(args)
    plt.show()

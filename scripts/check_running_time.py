import matplotlib.pyplot as plt
import sys

def read_log_file(filename):
    try:
        cont = None
        with open(filename, "r") as f:
            cont = f.readlines()

        core_update_time = []
        for line in cont:
            if line.find("Core_Update") != -1:
                core_update_time.append(float(line.split()[3]))
        return core_update_time

    except FileExistsError:
        print("file %s is not exist" % filename)
        return None

if __name__ =="__main__":
    # print("succ")
    args = sys.argv[1:] # file list

    # check file valid
    for filename in args:
        core_update_time = read_log_file(filename)
        plt.subplot(1, 1, 1)
        plt.plot(core_update_time, label=filename + " itr/h")
        plt.legend()
        # plt.legend([filename + " train_ret", filename + " test_ret", filename + " timer", ])
    # plt.legend(args)
    plt.show()

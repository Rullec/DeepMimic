import matplotlib.pyplot as plt
import sys

def read_log_file(filename):
    try:
        cont = None
        with open(filename, "r") as f:
            cont = f.readlines()

        wall_time_bucket = [0] * 48 * 6
        for line in cont:
            if line.find("Wall") != -1:
                t = float(line.split()[3]) * 6
                wall_time_bucket[int(t)] += 1
        while wall_time_bucket[-1] == 0: wall_time_bucket.pop()
        return wall_time_bucket

    except FileExistsError:
        print("file %s is not exist" % filename)
        return None

if __name__ =="__main__":
    # print("succ")
    args = sys.argv[1:] # file list

    # check file valid
    for filename in args:
        wall_time_bucket = read_log_file(filename)
        plt.subplot(1, 1, 1)
        plt.plot(wall_time_bucket, label=filename + " itr/h")
        plt.legend()
        # plt.legend([filename + " train_ret", filename + " test_ret", filename + " timer", ])
    # plt.legend(args)
    plt.show()

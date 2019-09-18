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
        cmd = "cat %s | grep -i train_return | awk '{print $4}' | grep -v =" % filename
        ret = subprocess.getoutput(cmd).split()
        ret = [float(i) for i in ret]

        cmd = "cat %s | grep -i timer | awk '{print $7}' | grep -v [a-z]" % filename
        time_series = subprocess.getoutput(cmd).split()
        gap_divide = int(len(time_series) / len(ret))
        times = []
        for i, cont in enumerate(time_series):
            if i % gap_divide ==0:
                print(cont)
                times.append(float(cont))

        return ret, times
        # with open(filename) as f:
        #     cont = f.readlines()
        #     if len(cont) == 1:
        #         cont = cont[0].split()
        #         # print(cont)
            
        #     num_list = [float(i) for i in cont]
            
        #     return num_list

    except FileExistsError:
        print("file %s is not exist" % filename)
        return None

if __name__ =="__main__":
    # print("succ")
    args = sys.argv[1:] # file list
    
    # check file valid
    for filename in args:
        cont, times = read_log_file(filename)
        plt.plot(cont)
        plt.plot(times)
        plt.legend([filename + " reward", filename + " timer"])
    # plt.legend(args)
    plt.show()

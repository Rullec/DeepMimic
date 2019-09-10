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
        output = subprocess.getoutput(cmd).split()
        output = [float(i) for i in output]
        return output
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
        cont = read_log_file(filename)
        plt.plot(cont)
    plt.legend(args)
    plt.show()

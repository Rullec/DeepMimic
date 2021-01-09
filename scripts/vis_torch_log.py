import matplotlib.pyplot as plt
import sys
import os
if __name__ == "__main__":
    file = sys.argv[1]
    assert os.path.exists(file) == True
    with open(file) as f:
        cont = f.readlines()
    rew_lst = []
    lr_lst = []
    for line in cont:
        if -1 != line.find("mean reward"):
            num = line.split()[-1]
            num = float(num)
            if num > 1e-5:
                rew_lst.append(float(num))
        if -1 != line.find("lr"):
            num = line.split()[-1]
            lr_lst.append(float(num))
    # print(lr_lst)
    plt.subplot(1, 2, 1)
    plt.plot(rew_lst)
    plt.title(f"{file} reward")
    plt.subplot(1, 2, 2)
    plt.plot(lr_lst)
    plt.title("lr")
    plt.show()

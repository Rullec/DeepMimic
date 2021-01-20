import matplotlib.pyplot as plt
with open("log") as f:
    a0_lst = []
    a1_lst = []
    drda0_lst = []
    drda1_lst = []
    for line in f.readlines():
        if line.find("[debug] action") != -1:
            a, b = line.split()[3:5]
            drda0, drda1 = line.split()[7:9]
            print(drda0)
            try:
                a = float(a[1:])
                b = float(b[:-1])
                drda0 = float(drda0[1:])
                drda1 = float(drda1[:-1])
                a0_lst.append(a)
                a1_lst.append(b)
                drda0_lst.append(drda0)
                drda1_lst.append(drda1)
            except:
                continue
    plt.plot(a0_lst)
    plt.plot(a1_lst)
    plt.plot(drda0_lst)
    plt.plot(drda1_lst)
    plt.legend(["a0", "a1", "drda0", "drda1"])
    plt.show()
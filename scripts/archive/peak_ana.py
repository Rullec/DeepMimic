import numpy as np
import os
import matplotlib.pyplot as plt
from aa_theta_dist_visualizer import mGuranty, paint_sample
res = os.popen("cat ../log | grep -i joint | grep -i theta").readlines()

paint_sample = 1
x_axis_guranty = [0 for i in range(mGuranty)]
y_theta_val = [0 for i in range(mGuranty)]
theta_lst = []
for id, cur_line in enumerate(res):
    if id % paint_sample == 0:
        phase = float(cur_line.split()[4])
        theta = float(cur_line.split()[7])
        # y_theta_val[int(phase *mGuranty)] += theta > 0 - theta < 0
        y_theta_val[int(phase * mGuranty)] += theta
plt.plot(y_theta_val)
plt.show()

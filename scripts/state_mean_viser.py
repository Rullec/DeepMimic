import matplotlib.pyplot as plt
import re
import numpy as np

with open("log") as f:
    cont = f.read()
    # for i in f.readlines():
    #     cont = cont + i
# cont = '''[stat] state std [0.24332 0.00107 0.      0.00003 0.00171 0.00004 0.00565 0.      0.
#  0.      0.00061 0.00235 0.0009  0.00551 0.      0.      0.      0.00131
#  0.00365 0.00174 0.0003  0.00002 0.      0.      0.00186 0.00348 0.00528
#  0.00538 0.      0.      0.      0.0006  0.00233 0.00542 0.00093 0.00026
#  0.00004 0.      0.00128 0.00364 0.00156 0.00027 0.00001 0.      0.
#  0.00155 0.00344 0.00213 0.00211 0.      0.      0.      0.0132  0.04985
#  0.09888 0.      0.      0.      0.00817 0.01693 0.10554 0.      0.
#  0.      0.00232 0.00671 0.04334 0.      0.      0.      0.00001 0.
#  0.      0.      0.      0.      0.00817 0.01694 0.10542 0.      0.
#  0.      0.00232 0.0067  0.04325 0.      0.      0.      0.00001 0.
#  0.      0.      0.     ]
# [stat] action mean [-0.24681'''


def parse_np_prefix(prefix, cont):
    pat = re.compile(rf"(?<={prefix})[\.\s\d]*(?=\])", re.M | re.S)

    res_lst = []
    for l in pat.findall(cont):
        res_lst.append([float(i) for i in l.split()])
    return res_lst

state_std = np.array(parse_np_prefix(r"state std \[", cont))
action_std = np.array(parse_np_prefix(r"action std \[", cont))
drda_std = np.array(parse_np_prefix(r"drda std \[", cont))
state_std_norm = np.linalg.norm(state_std, axis=1)
action_std_norm = np.linalg.norm(action_std, axis=1)
drda_std_norm = np.linalg.norm(drda_std, axis=1)

plt.subplot(1, 3, 1)
plt.plot(state_std_norm)
plt.title("state_std_norm")
plt.subplot(1, 3, 2)
plt.plot(action_std_norm)
plt.title("action_std_norm")
plt.subplot(1, 3, 3)
plt.plot(drda_std_norm)
plt.title("drda_std_norm")


# print(res.shape)
plt.tight_layout()
plt.show()
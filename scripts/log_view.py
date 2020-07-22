import numpy as np

A = np.genfromtxt('agent0_log.txt',delimiter='\t',dtype=None, names=True, encoding=None)
print(A)
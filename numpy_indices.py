import numpy as np


a = np.array([0, 0, 0, 0])
i = np.array([1, 2])

a[i] += 1
print(a)

b = np.array([0, 0, 0, 0])
j = np.array([1, 1])
b[j] += 1
print(b)

c = [0, 0, 0]
k = [1, 2]
try:
    c[k] += 1
    print(c)
except TypeError as e:
    print(f"{c}{k}{e}")

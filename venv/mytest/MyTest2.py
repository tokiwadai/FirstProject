from typing import Tuple

import numpy as np


box1: Tuple[int, int, int, int] = (2, 1, 4, 3)
box2 = (1, 2, 3, 4)
print("box1: " + str(box1))
print("box1: " + str(box1[1]))
print("box1: " + str(max(1, 2)))


a = np.zeros((10,2))
print("a: " + str(np.sum(a)))
print("a.shape: " + str(a.shape))
print("a.T.shape: " + str(a.T.shape))
print("a.size: " + str(a.size))
print("a.shape[0]: " + str(a.shape[0]))
print("a.shape[1]: " + str(a.shape[1]))

a = np.zeros((10,2,7))
b = a[1,:]
print("b.shape: " + str(b.shape))

for i in range(0 ,10):
    print("i: " + str(i))

x = np.random.randn(4, 3, 3, 2)
y = x[1,:]


print(str(y.shape))

#print(str(x))
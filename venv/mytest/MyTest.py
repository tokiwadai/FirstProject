import numpy as np


string=""
string1=""
string2=""
string3=""
string4=""
string5=""
string6=""
string7=""
string8=""
string9=""

for i in range(0 ,4):
    cm = ", "
    if(i == 3):
       cm = ""
    string += (str(410+i) + cm)
    string1 += (str(510 + i) + cm)
    string2 += (str(420 + i) + cm)
    string3 += (str(520 + i) + cm)
    string4 += (str(700 + i) + cm)
    string5 += (str(800 + i) + cm)
    string6 += (str(710 + i) + cm)
    string7 += (str(810 + i) + cm)
    string8 += (str(720 + i) + cm)
    string9 += (str(820 + i) + cm)

print(string)
print(string1)
print(string2)
print(string3)
print(string4)
print(string5)
print(string6)
print(string7)
print(string8)
print(string9)


print(str(np.sqrt(4)))

for x in reversed(range(5)):
    print("x: " + str(x))

for x in (range(5)):
    print("x: " + str(x))

print("hello world")
A = np.random.randn(4,3)
B = np.sum(A, axis = 1, keepdims = True)
print("B.shape: " + str(B.shape))


a = np.zeros((10,2))
print("a: " + str(np.sum(a)))
print("a.shape: " + str(a.shape))
print("a.T.shape: " + str(a.T.shape))
print(a.size)
print((a.shape[0]))

b = a.reshape(a.shape[0] * a.shape[1],-1)
##print(b)

p = 1
c = np.zeros((3,3,2))
for x in range(0, c.shape[0]):
    for y in range(0, c.shape[1]):
        for z in range(0, c.shape[2]):
            c[x, y, z] = p
            p += 1

print("c:" + str(c) + ", " + str(c[1, 2, 1]))

c_reshape = c.reshape(c.shape[0], -1)
print("c_reshape: \n" + str(c_reshape))

c_reshapeT = c.reshape(c.shape[0], -1).T
print("c_reshapeT: \n" + str(c_reshapeT))

c_reshape0 = c.reshape(c.shape[0] * c.shape[1] * c.shape[2], -1).T
print("c_reshape0: \n" + str(c_reshape0))


c_reshape1 = c.reshape(c.shape[0] * c.shape[1] * c.shape[2], 1).T
print("c_reshape1: \n" + str(c_reshape1))


a = np.random.randn(2, 3) # a.shape = (2, 3)
b = np.random.randn(2, 1) # b.shape = (2, 1)
c = a + b
print ("c.shape1: " + str(c.shape))

a = np.random.randn(12288, 150) # a.shape = (12288, 150)
b = np.random.randn(150, 45) # b.shape = (150, 45)
c = np.dot(a,b)
print ("c.shape3: " + str(c.shape))

a = np.random.randn(4, 3) # a.shape = (4, 3)
b = np.random.randn(3, 2) # b.shape = (3, 2)
c = a*b
print ("c.shape2: " + str(c.shape))



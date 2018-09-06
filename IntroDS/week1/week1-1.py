import numpy as np
import sklearn

#Matrices
a = list(range(0,10000000))
b = a

def add_with_for(a,b):
    c = []
    for i in range(len(a)):
        c.append(a[i]+b[i])
    return c

c = add_with_for(a,b)

npa = np.array(range(0,10000000))
npb = npa
npc = npa+npb

matrix1 = np.array([range(0,100)]).reshape((10,10))
print(matrix1)

matrix2 = np.tile([0.0,1.0], 50).reshape((10,10,))
print(matrix2)

tmp = np.eye(10)
matrix3 = 1-tmp
print(matrix3)

matrix4 = np.flipud(matrix3)
print(matrix4)

print(np.linalg.det(matrix3)*np.linalg.det(matrix4))
print(np.linalg.det(np.matmul(matrix3,matrix4)))

#Boston
from sklearn.datasets import load_boston
boston = load_boston()
#print(boston.data.shape)
crim = np.where(boston.data[:,0] > 1)
print(crim)
print(crim[0].size)

pttr = boston.data[:,10]
tmp = np.where(np.logical_and(np.less(pttr,18), np.greater(pttr,16)))
print(tmp)
print(np.asarray(tmp).size)

nitro = boston.data[:,4]
nox = np.where(boston.target > 25)
mean = np.mean(nitro[nox])
print(mean)

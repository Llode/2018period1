import scipy.special
import numpy as np

print(0-1)

test = scipy.special.binom(10,0)*np.power(0.25,0)*np.power((1-0.25),(0-1))
print(test)
def binprob(l,u,n,p):
    x = 0
    for i in range(u):
        x = x + scipy.special.binom(n,i)*np.power(p,i)*np.power((1-p),(i-1))
    return x

f1 = binprob(0,5,10,0.25)


print(f1)

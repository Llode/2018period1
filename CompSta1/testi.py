#%%
import numpy as np
import numpy.random as npr
import pandas as pd
import scipy.stats as sst
import matplotlib.pyplot as plt

npr.seed(1345)
# Load the data set
babies_full = pd.read_csv("http://www.helsinki.fi/~ahonkela/teaching/compstats1/babies.txt", sep='\t')

N=1000

y = babies_full['gestation'].values
x = babies_full['age'].values
var = 1
epsilon = npr.normal(0,var)

def loglh(x,y,n,a,b,C,var):
    res = 0
    for i in range(n):
        res = res + (-(1/2*var)*(y[i]-a-b*x[i])**2 + C)
    return res

def b_hat(x,y,a):
    xmean = np.mean(x)
    ymean = np.mean(y)
    x2 = x-xmean
    y2 = y-ymean
    return (x2*y2)/(x2**2) * a

plt.scatter(x,y)
plt.plot(x,y)
plt.show()

print(b_hat(x,y,1))


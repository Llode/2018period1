#%%
import numpy as np
import numpy.random as npr
import pandas as pd
import scipy.stats as sst
import matplotlib.pyplot as plt
import seaborn as sns


npr.seed(1345)
# Load the data set
babies_full = pd.read_csv("http://www.helsinki.fi/~ahonkela/teaching/compstats1/babies.txt", sep='\t')
bwt = babies_full['bwt'].values
age = babies_full['age'].values
y = bwt
x = age

X=np.column_stack((y,x))
print(X.shape[1])

def kh_m(x1,x2):
    first = (2*np.pi)**(-1)
    norm = x1**2 + x2**2
    second = np.exp((-(norm))/2)
    return first*second

def kernel_density(xgrid,ygrid, x, y,h):
    z = np.zeros((len(ygrid), len(xgrid)))

    for i in range(len(ygrid)):
        for j in range(len(xgrid)):
            z[i][j] = np.mean(kh_m(((ygrid[i] - y)/ h),((xgrid[j] - x)/ h))) / h**2
    return z

tx = np.linspace(0,x.max(), x.max())
ty = np.linspace(0,y.max(), y.max())
kdens = kernel_density(tx,ty,x,y,5)
print('estimated density at h=5: ', kdens[120][25])


def cross_validate_density(d):
    hs = np.linspace(1.0,10.0,30)
    logls = np.zeros(len(hs))
    for j in range(d.shape[0]):
        for k in range(d.shape[1]):
            for i in range(len(hs)):
                logls[i] += np.sum(np.log(kernel_density(d[k],d[j], np.delete(d, k), np.delete(d,j), hs[i])))
    return (hs, logls)
hs, logls = cross_validate_density(X)

h_opt = hs[np.argmax(logls)]
print("Optimal h:", h_opt)

kdens2 = kernel_density(tx,ty,x,y,h_opt)
print('estimated density at optimal h: ', kdens2[120][25])

plt.hist2d(bwt,age)
plt.show()
sns.kdeplot(bwt, age, kernel='gau')
plt.show()
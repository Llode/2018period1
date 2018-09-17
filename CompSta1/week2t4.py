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

def Kh(x1,x2):
    first = 1/(2*np.pi*h**2)
    second = np.exp(-(x1**2 + x2**2)/(2*h**2))
    return first*second

def kernel_density(t, x1,x2, h):
    y = np.zeros(len(t))
    for i in range(len(t)):
        y[i] = np.mean(Kh(((t[i] - x1)/ h),((t[i] - x2)/ h))) / h
    return y

def LOO(x,y):
    t = np.linspace(1.0, 10.0, 30)
    logls = np.zeros(len(t))
    for ii in range(len(x)):
        for i in range(len(t)):
            logls[i] += np.sum(np.log(kernel_density([y[ii]], np.delete(x,ii), np.delete(y,ii), t[i])))

    return(t, logls)

t = np.linspace(1.0, 175,300)
h=5

kernel_h = Kh(age,bwt)
density = kernel_density(t,bwt, age, h)
density_q = kernel_density(t,120,25,h)
denq = np.mean(density_q)
print('density at bwt=120 age=25 ', denq)

hs, logls = LOO(bwt,age)
plt.hist(bwt, 30, normed=True)
plt.show()

h_opt = hs[np.argmax(logls)]
print("Optimal h:", h_opt)

t = np.linspace(1.0, 10.0, 30)
density_opt = kernel_density(t, bwt,age, h_opt)
denq_op = np.mean(density_opt)
print('density at bwt=120 age=25 ', denq_op)

plt.hist2d(bwt,age)
plt.show()
sns.kdeplot(bwt, age, kernel='gau')
plt.show()

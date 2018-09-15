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

t = np.linspace(1.0, 150.0, 300)
matrix = np.column_stack((bwt, age))
h=5


plt.plot(t, kernel_density(t, bwt,age, h), label='h=5.0')


#plt.hist2d(bwt,age)
Kres = Kh(age,bwt)
matrix = np.column_stack((bwt, age, Kres))
print(len(Kres), np.max(Kres))
print(Kres)
print(Kh(25,120))
#sns.kdeplot(bwt, age, kernel='gau')



def LOO(x,y):
    t = np.linspace(1.0, 10.0, 30)
    logls = np.zeros(len(t))
    for ii in range(len(x)):
        for i in range(len(t)):
            logls[i] += np.sum(np.log(kernel_density([y[ii]], np.delete(x,ii), np.delete(y,ii), t[i])))

    return(t, logls)

hs, logls = LOO(bwt,age)
plt.hist(bwt, 30, normed=True)
h_opt = hs[np.argmax(logls)]
print("Optimal h:", h_opt)
t = np.linspace(1.0, 10.0, 30)
plt.plot(t, kernel_density(t, bwt,age, h_opt))
plt.show()
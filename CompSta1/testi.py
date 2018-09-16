#%%
import numpy as np
import numpy.random as npr
import pandas as pd
import scipy.stats as sst
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

npr.seed(1345)
# Load the data set
babies_full = pd.read_csv("http://www.helsinki.fi/~ahonkela/teaching/compstats1/babies.txt", sep='\t')

N=1000

y = babies_full['gestation'].values
x = babies_full['age'].values

xmean = np.mean(x)
ymean = np.mean(y)
#sns.regplot(x,y)

#print(sm.OLS(y,x).fit().summary())

var = 1
def epsilon():
    return npr.normal(0,var)


def loglh(x,y,n,a,b,C,var):
    res = 0
    for i in range(n):
        res = res + (-(1/2*var)*(y[i]-a-b*x[i])**2 + C)
    return res

def a_hat(ymean, xmean, bhat):
    return ymean - bhat*xmean

def b_hat(x,y,xmean,ymean):
    first = 0
    second = 0
    for i in range(len(x)):
        first = first + (x[i]-xmean)*(y[i]-ymean)
        second = second + (x[i]-xmean)**2

    return first/second

bhat = b_hat(x,y,xmean,ymean)
print('bhat ', bhat)
ahat = a_hat(ymean,xmean, bhat)
print('ahat ', ahat)

plt.scatter(x,y)
plt.plot(x,ahat+bhat*x +epsilon())
plt.show()
sns.regplot(x,y)
plt.show()

def bootstrapMeans(data, n):
    bootstrap_means = np.array([np.mean(npr.choice(data, replace=True, size=n)) for i in range(N)])
    print('bootstrap interval:', np.percentile(bootstrap_means, [2.5, 97.5]))
    ival1, ival2 = np.percentile(bootstrap_means, [2.5, 97.5])
    ival_width = abs(ival1-ival2)
    print('interval width: ', ival_width)

bootstrapMeans(ahat+bhat*x +epsilon(), N)
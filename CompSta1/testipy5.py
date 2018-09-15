import pandas as pd
import numpy as np
import numpy.random as npr
import scipy.stats as sst

npr.seed(1345)
# Load the data set
babies_full = pd.read_csv("http://www.helsinki.fi/~ahonkela/teaching/compstats1/babies.txt", sep='\t')

# Pick a subset
babies1 = babies_full.iloc[(babies_full['gestation']>=273).values]

babiesYoung = babies1.query('age<26')
babiesOld = babies1.query('age>=26')

#print(babies1.head(10))

X1 = babiesYoung['bwt'].mean()
X2 = babiesOld['bwt'].mean()
t = np.abs(X1-X2)
s1 = np.var(babiesYoung['bwt'])
s2 = np.var(babiesOld['bwt'])
N1 = len(babiesYoung.index)
N2 = len(babiesOld.index)

def sTest(X1,X2,s1,s2,N1,N2):
    return (X1-X2)/np.sqrt((s1/N1)+(s2/N2))

tX, pX = sst.ttest_ind(babiesYoung['bwt'], babiesOld['bwt'], equal_var=False)
tXf = sTest(X1,X2,s1,s2,N1,N2)
print('pFunktio ', pX)
print('t oma ', tXf)
df = N1+N2 -2
pXf = sst.t.cdf(tXf,df=df)
t_val = sst.t.ppf([pXf], df)
print('t_val ', t_val)
print('pXf ', pXf)



#vectors
Ismoker = babies1.query('smoke==1')
Inonsmoker = babies1.query('smoke==0')

smokermean = Ismoker.bwt.mean()
nonsmokermean = Inonsmoker.bwt.mean()
truediff_smoke = np.float64(np.abs(smokermean - nonsmokermean))


tX, pX = sst.ttest_ind(Ismoker['bwt'], Inonsmoker['bwt'], equal_var=False)
print(tX, pX)
print('smoker mean ', truediff_smoke)


N_perm = 50
meandiffs = np.zeros(N_perm)
numsmokers = sum(Ismoker.bwt)

for i in range(N_perm):
    z = npr.permutation(len(Ismoker))
    zz =  npr.permutation(len(Inonsmoker))

    #meandiffs[i] = np.float64(np.abs(z['bwt'].mean() - zz['bwt'].mean()))

# print(meandiffs)
# print(len(meandiffs))
# print(np.sum(truediff_smoke <= meandiffs))
# print('p-value:', np.float64((np.sum(truediff_smoke < meandiffs)+1)/(len(meandiffs)+1)))

# arr = range(0,8)
# print(truediff_smoke < meandiffs)
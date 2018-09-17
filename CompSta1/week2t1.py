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

smoker = babies1.query('smoke>0')
nonSmoker = babies1.query('smoke<1')

#print(babies1.head(10))

X1 = babiesYoung['bwt'].mean()
X2 = babiesOld['bwt'].mean()

s1 = np.var(babiesYoung['bwt'])
s2 = np.var(babiesOld['bwt'])
N1 = len(babiesYoung.index)
N2 = len(babiesOld.index)

truediff = np.abs(X1-X2)

def sTest(X1,X2,s1,s2,N1,N2):
    return (X1-X2)/np.sqrt((s1/N1)+(s2/N2))

tX, pX = sst.ttest_ind(babiesYoung['bwt'], babiesOld['bwt'], equal_var=False)


tXf = sTest(X1,X2,s1,s2,N1,N2)
print('P-value ', pX)
print('T-value ', tX)
print('bm', tXf )
print('t oma ', tXf)
print('true abs ', truediff)
df = N1+N2 -2
pXf = sst.t.cdf(tXf,df=df)
t_val = sst.t.ppf([pXf], df)
print('t_val_check ', t_val)
print('pXf ', pXf)

oldSmoker = babiesOld.query('smoke==1')['bwt']
oldNonsmoker = babiesOld.query('smoke==0')['bwt']
youngSmoker = babiesYoung.query('smoke==1')['bwt']
youngNonsmoker = babiesYoung.query('smoke==0')['bwt']

N_perm = 50000


truediff_smoke = truediff

print('smoker mean ', truediff_smoke)
print(oldNonsmoker.mean(), oldNonsmoker.mean(), youngNonsmoker.mean(), youngSmoker.mean())
meandiffs = np.zeros(N_perm)
for i in range(N_perm):
    z1 = npr.permutation(oldSmoker)
    z2 = npr.permutation(youngSmoker)
    z3 = npr.permutation(oldNonsmoker)
    z4 = npr.permutation(youngNonsmoker)
    diff1 = np.concatenate((z1,z3))
    diff2 = np.concatenate((z2,z4))
    meandiffs[i] = np.abs(diff1.mean() - diff2.mean())


#print(meandiffs)
print('p-value:', (np.sum(truediff_smoke <= meandiffs)+1)/(len(meandiffs)+1))
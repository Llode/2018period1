import pandas as pd
import numpy as np
import numpy.random as npr
import scipy.stats as sst

""" npr.seed(1345)
# Load the data set
babies_full = pd.read_csv("http://www.helsinki.fi/~ahonkela/teaching/compstats1/babies.txt", sep='\t')
import pandas as pd
import numpy as np
import numpy.random as npr """

# load the data from CSV file using pandas
babies_full = pd.read_csv("http://www.helsinki.fi/~ahonkela/teaching/compstats1/babies.txt", sep='\t')
fram = babies_full.iloc[(babies_full['gestation']>=273).values]

Ismoker = (fram['smoke'] == 1).values
Inonsmoker = (fram['smoke'] == 0).values
highAge = (fram['age'] >= 26).values
lowAge = (fram['age'] < 26).values

nonsmokermean = np.mean(fram['bwt'].values[Inonsmoker])
smokermean = np.mean(fram['bwt'].values[Ismoker])
lowagemean = np.mean(fram['bwt'].values[lowAge])
highagemean = np.mean(fram['bwt'].values[highAge])
print(nonsmokermean, smokermean)

IhighAge = np.where(highAge)[0]
IlowAge = np.where(lowAge)[0]
IIsmoke = np.where(Ismoker)[0]
IInonsmoke = np.where(Inonsmoker)[0]

numrepeats = 50000
mean_differences = np.zeros(numrepeats)
npr.seed(34234234)
print(Ismoker.shape, highAge.shape)
numhighAgesmokers = np.sum(Ismoker & highAge)
print(numhighAgesmokers)
numlowAgesmokers = np.sum(Ismoker & lowAge)

for i in range(numrepeats):
    I1 = npr.permutation(IhighAge)
    I2 = npr.permutation(IlowAge)
    permmean1 = np.mean(np.concatenate((fram['bwt'].values[I1[0:numhighAgesmokers]],
                                        fram['bwt'].values[I2[0:numlowAgesmokers]])))
    permmean2 = np.mean(np.concatenate((fram['bwt'].values[I1[numhighAgesmokers:]],
                                        fram['bwt'].values[I2[numlowAgesmokers:]])))
    #print(permmean2, permmean1)
    mean_differences[i] = np.abs(permmean1 - permmean2)

print((np.sum(np.abs(lowagemean - highagemean) <= mean_differences) + 1)/(len(mean_differences)+1))
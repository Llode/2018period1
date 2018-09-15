nonsmokermean = np.mean(babies1.iloc[Inonsmoker]['bwt'].values)

truediff_smoke = np.float64(np.abs(smokermean - nonsmokermean))


tX, pX = sst.ttest_ind(smoker['bwt'], nonSmoker['bwt'], equal_var=False)
print(tX, pX)
print('smoker mean ', truediff_smoke)

meandiffs = np.zeros(N_perm)
for i in range(N_perm):
    z = npr.permutation(len(Ismoker))
    zz =  npr.permutation(len(Inonsmoker))
    print(np.abs(smoker['bwt'][z].mean()))
    #meandiffs[i] = np.float64(np.abs(z['bwt'].mean() - zz['bwt'].mean()))

# print(meandiffs)
# print(len(meandiffs))
# print(np.sum(truediff_smoke <= meandiffs))
# print('p-value:', np.float64((np.sum(truediff_smoke < meandiffs)+1)/(len(meandiffs)+1)))

# arr = range(0,8)
# print(truediff_smoke < meandiffs)
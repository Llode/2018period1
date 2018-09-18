import numpy as np
import numpy.random as npr
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

# Load the data set
babies_full = pd.read_csv("http://www.helsinki.fi/~ahonkela/teaching/compstats1/babies.txt", sep='\t')

# Pick a subset
babies5 = babies_full.loc[(babies_full['age']>=30).values,:]
#print(babies5.head(10))
y = babies5['bwt'].values
x = babies5['gestation'].values

ymean = np.mean(y)
xmean = np.mean(x)

var = 1
def epsilon():
    return npr.normal(0,var)

def RMSE(pred, target):
    return np.sqrt(np.mean((target-pred)**2))

def a_hat(ymean, xmean, bhat):
    return ymean - bhat*xmean

def b_hat(x,y,xmean,ymean):
    first = 0
    second = 0
    for i in range(len(x)):
        first = first + (x[i]-xmean)*(y[i]-ymean)
        second = second + (x[i]-xmean)**2

    return first/second

def predict(x,y):
    ymean = np.mean(y)
    xmean = np.mean(x)
    bhat = b_hat(x,y,xmean,ymean)
    ahat = a_hat(ymean,xmean, bhat)
    return ahat+bhat*x +epsilon()

def predict_coefs(x,y):
    ymean = np.mean(y)
    xmean = np.mean(x)
    bhat = b_hat(x,y,xmean,ymean)
    ahat = a_hat(ymean,xmean, bhat)
    return ahat, bhat

def calc_y(x, a_hat, b_hat):
    return ahat+bhat*x +epsilon()


bhat = b_hat(x,y,xmean,ymean)
ahat = a_hat(ymean,xmean, bhat)
print('bhat ', bhat)
print('ahat ', ahat)

prediction = ahat+bhat*x
err = RMSE(prediction, y)
print('RMSE ', err)

rms = np.sqrt(mean_squared_error(y, prediction))
print('scipy mse ', rms)

#check that linreg is valid
plt.scatter(x,y)
plt.plot(x,prediction)
plt.show()
sns.regplot(x,y)
plt.show()


k=20
sum=0
arr = np.transpose(np.asarray([x,y]))

kf = KFold(n_splits=k)
for train, test in kf.split(arr):
    test_data = arr[test]
    train_data = arr[train]
    ahat, bhat = predict_coefs(train_data[:,0], train_data[:,1])
    pr = calc_y(test_data[:,0], ahat, bhat)
    sum = sum + (RMSE(pr, test_data[:,1]))**2
avg_MSE = np.sqrt(sum/k)
print('kFold RMSE ', avg_MSE)



###  MULTIVARIATE  ###
gest = babies5['gestation'].values
age = babies5['age'].values
weight = babies5['weight'].values
bwt = babies5['bwt'].values

X = np.column_stack((gest,age,weight))
X1 = np.append(X, np.ones((len(X),1)), axis =1)


sns.regplot(gest,age,weight)
plt.show()

def b_hat4(X,y):
    Xt = np.transpose(X)
    first = np.matmul(Xt,X)
    # invXtX = np.linalg.inv(np.linalg.cholesky(first))
    # invXtX = np.dot(np.transpose(invXtX),invXtX)
    invXtX = np.linalg.inv(first)
    multX = np.matmul(invXtX, Xt)
    return np.matmul(multX,y)

def multiy(X, bhat):
    return np.matmul(X, bhat)

bhat4 = b_hat4(X1, bwt)

print('bhat ', bhat4)

yyy = multiy(X1, bhat4)
print(yyy.shape)
""" plt.scatter(gest, age, weight)
plt.plot(X,yyy)
plt.show() """

sum4 = 0
kf2 = KFold(n_splits=len(weight))
for train, test in kf2.split(X1):
    test_data4 = X1[test]
    train_data4 = X1[train]
    bhat4 = b_hat4(train_data4,train_data4[:,2])
    yhat4 = multiy(test_data4,bhat4)
    sum4 += (RMSE(yhat4, test_data4[:,2]))**2
avg_MSE4 = np.sqrt(sum4/len(weight)**2)
print('LOO RMSE ', avg_MSE4)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x1 = X1[:,0]
x2 = X1[:,1]
x3 = X1[:,2]

ax.plot_surface(x1, x2,
                yyy.reshape(x1.shape),
                rstride=1,
                cstride=1,
                color='None',
                alpha = 0.4)

ax.scatter3D(x1, x2, x3, c=x3, cmap='Greens');
ax.plot3D(bwt, yyy)
plt.show()

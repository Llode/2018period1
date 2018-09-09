import scipy.special
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.stats import uniform
from sklearn.preprocessing import normalize

npr.seed(2349678)

def RejectionSampler(f_pdf, g_pdf, g_sample, M, N):
    i = 0
    x = np.zeros(N)
    while i < N:
        x_prop = g_sample()
        u = npr.uniform(0,1)
        if(u*M*g_pdf(x_prop) < f_pdf(x_prop)):
            x[i] = x_prop
            i += 1
    return x

def laplace(x):
    return 0.5*np.exp(-np.abs(x))

def normpdf(x):
    return 1/np.sqrt(2*np.pi)*np.exp(-0.5*x**2)

def p(x):
    return laplace(x)
 
def q(x):
    return normpdf(x)

def sample():
    return npr.randn()

N=10000
linrange = 4
t = np.linspace(-linrange,linrange, N)
m = np.linspace(-2,2,N)
M=np.max(p(t)/q(t))
#M=3


#print('sample ' + str(sample()))
#print('psample ' + str(p(sample())))
res = RejectionSampler(p, q, sample, M, N)
print(np.mean(res**4))

plt.figure(11)
plt.subplot(211)
plt.plot(t,p(t), 'k')
plt.plot(t,M*q(t), 'r')
plt.show()

fig, ax = plt.subplots(1, 2)
ax[0].plot(t, p(t) / (M*q(t)))
ax[0].set_title('$f(x) / (M \cdot g(x))$')
ax[1].hist(res, 100, normed=True)
ax[1].plot(t, p(t), 'g')
ax[1].set_title('samples')
plt.show()
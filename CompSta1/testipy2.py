import scipy.special
import numpy as np

d=2
mu = mu = np.array([0,0])[np.newaxis]
x0 = mu
x1 = np.array([1,1])[np.newaxis]
# sigma = np.array([np.power(2,2),2*p,2*p,1]).reshape(d,d)

def initSigma(p, d):
    sigma = np.array([np.power(2,2),2*p,2*p,1]).reshape(d,d)
    #print('sigma')
    #print(sigma)
    return sigma

def logdetsig(L, d):
    sum = 0
    for j in range(d):
        sum = sum+np.log(L[j,j])
    #print('determinantti')
    #print(2*sum)
    return 2*sum

def calcLnd(x, mu, d, L, det):
    arr = (x-mu).T
    #print('x-mu')
    #print(arr)
    z =  scipy.linalg.solve_triangular(L, arr, lower=True)
    z = np.linalg.norm(z)
    #print('solved')
    #print(z)
    res = -0.5*d*np.log(2*np.pi) - 0.5*det - 0.5*z
    return res

def LND(x, mu, p):
    sigma = initSigma(p,d)
    L = np.linalg.cholesky(sigma)
    det = logdetsig(L,d)
    res = calcLnd(x,mu,d,L,det)
    return res

i = LND(mu,mu, 0.8)
ii = LND(mu,mu,0.999)
iii = LND(x1,mu, 0.999)
iv = LND(x1,mu,-0.999)

print(i)
print(ii)
print(iii)
print(iv)

9007199254740991
print (stL(rnp.iinfo(np.float64).max))
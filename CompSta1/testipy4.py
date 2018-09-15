#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
d = pd.read_csv('http://www.helsinki.fi/~ahonkela/teaching/compstats1/toydata.txt').values
plt.hist(d, 30, normed=True)
def K_gauss(x):
    return 1/np.sqrt(2*np.pi)*np.exp(-0.5*x**2)
def kernel_density(t, x, h):
    y = np.zeros(len(t))
    for i in range(len(t)):
        y[i] = np.mean(K_gauss((t[i] - x)/ h)) / h
    return y
t = np.linspace(-2, 10, 100)
plt.plot(t, kernel_density(t, d, 3.0), label='h=3.0')
plt.plot(t, kernel_density(t, d, 1.0), label='h=1.0')
plt.plot(t, kernel_density(t, d, 0.3), label='h=0.3')
plt.plot(t, kernel_density(t, d, 0.1), label='h=0.1')
plt.legend()

def cross_validate_density(d):
    hs = np.linspace(0.1, 1.0, 10)
    logls = np.zeros(len(hs))
    for j in range(len(d)):
        for i in range(len(hs)):
            logls[i] += np.sum(np.log(kernel_density(d[j], np.delete(d, j), hs[i])))
            print(type(d[j]))
    return (hs, logls)
hs, logls = cross_validate_density(d)
plt.hist(d, 30, normed=True)
h_opt = hs[np.argmax(logls)]
print("Optimal h:", h_opt)

t = np.linspace(-2, 10, 100)
plt.plot(t, kernel_density(t, d, h_opt))
plt.show()
%matplotlib inline
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

x = np.float64(1)
while(x != np.inf):
    if(x+1 == np.inf):
        break
    x = x+1
print(x)
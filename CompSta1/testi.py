import numpy as np
import numpy.random as npr
def RejectionSampler(f_pdf, g_pdf, g_sample, M, N):
    # Returns N samples following pdf f_pdf() using proposal g(x)
    # with pdf g_pdf() that can be sampled by g_sample()
    # Requirement: f_pdf(x) <= M*g_pdf(x) for all x
    i = 0
    x = np.zeros(N)
    while i < N:
        x_prop = g_sample()
        u = npr.uniform(0, 1)
        if (u * M * g_pdf(x_prop)) < f_pdf(x_prop):
            # Accept the sample and record it
            x[i] = x_prop
            i += 1
    return x

import matplotlib.pyplot as plt
# Set the random seed
npr.seed(2349678)
# Define normal pdf
def normpdf(x, mu, sigma):
    return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2 / (2*sigma**2))
# Define target pdf as a mixture of two normals
def target_pdf(x):
    return 12*x*(1-x)**2
# Define the proposal pdf and a function to sample from it
def proposal_pdf(x):
    return 1
def sample_proposal():
    return npr.random(1)
# Define M

N = 100
t = np.linspace(0.0, 1.0, N)
t2 = np.linspace(0.0, 1.0,N)
M = np.max(target_pdf(t)/proposal_pdf(t))

mysample = RejectionSampler(target_pdf, proposal_pdf, sample_proposal, M, N)
fig, ax = plt.subplots(1, 2)
t = np.linspace(0.0, 1.0, N)
t2 = np.linspace(0.0, 1.0,N)
# Plot f(x) / (M * g(x)) to verify M is valid (the line should be below 1)
ax[0].plot(t2, target_pdf(t2) / (M*proposal_pdf(t2)))
ax[0].set_title('$f(x) / (M \cdot g(x))$')
ax[1].hist(mysample, N, normed=True)
ax[1].plot(t, target_pdf(t), 'g')
ax[1].set_title('samples')
plt.show()

print(np.mean(mysample**5))
# Cuomo et al. (2015) O(n) Numerical Scheme
# https://www.sciencedirect.com/science/article/pii/S1877050915010066
# https://www.sciencedirect.com/science/article/pii/S1746809416300192?via%3Dihub

# Pros
# - Removes baseline wander from  signals
# - Does not need to perform the Fourier Transform
# - Runs in time O(n)

# Cons
# - It requires to know the noise frequency a priori

#%%

import math
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('MACOSX')


def recursive_filter(s0, mu, sigma):
  """
  Given a noisy signal, and a known frequency of the noise component, it
  attenuates the noise in the interval [mu-sigma, mu+sigma] Hz.
  @param s0: Input noisy signal. Should be a 1D np.array
  @param @mu: Central frequency of the known noise component (Hz)
  @param @sigma: Superior and inferior margin of the noise frequency (Hz)
  """

  tau = 1/sf  # discretization stepsize can be equal to sampling frequency
  n = np.size(s0)

  # Line 1: Coefficients
  b0 = 1
  b1 = -2 * math.cos(mu*tau)
  b2 = 1
  a1 = 2 * math.exp(-math.sqrt(2) * sigma * tau) * math.cos(mu * tau)
  a2 = -math.exp(-2 * math.sqrt(2) * sigma * tau)

  # Line 2: Initialize ph and border conditions
  ph = np.empty((n+2))
  ph[0] = ((b0 + b1 + b2) / (1 - a1 - a2)) * s0[0]
  ph[1] = ph[0]

  # Lines 3-4-5: Compute ph (forward phase)
  for j in range (2, n, 1):
    ph[j] = b0 * s0[j] + b1 * s0[j-1] + b2 * s0[j-2] + a1 * ph[j-1] + a2 * ph[j-2]

  # Line 6: Initialize sh and border conditions
  sh = np.empty((n+2))
  sh[n] = ((b0 + b1 + b2) / (1- a1 - a2)) * ph[n+1]
  sh[n-1] = sh[n]

  # Lines 7-8-9: Compute sh (backward phase)
  for j in range (n-2, 0, -1):
    sh[j] = b0 * ph[j] + b1 * ph[j+1] + b2 * ph[j+2] + a1 * sh[j+1] + a2 * sh[j+2]

  # Line 10
  return sh  # the filtered signal


def centralize(x):
    return x - np.mean(x)


filename = "/Users/jomy/Desktop/sinais HSM/Bitalinosignal_2021-03-29 18-22-31__2021-03-29 19-23-20.h5"
sf = 1000  # in Hz
signal = ((pd.read_hdf(filename))['ECG']).to_numpy()

# signal = signal[:300000]

n = np.size(signal)

samples = np.linspace(0, 5*60, n)
baselinewander1 = 200 * np.sin(2 * np.pi * 0.25 * samples)
samples = np.linspace(0, 5 * 60, n)
baselinewander2 = 200 * np.sin(2 * np.pi * 0.35 * samples)
noisy_signal = signal + baselinewander1 + baselinewander2

filtered_signal = recursive_filter(noisy_signal, 50, 0.5)
filtered_signal = recursive_filter(noisy_signal, 0.35, 0.05)
filtered_signal = recursive_filter(filtered_signal, 0.25, 0.05)

# Normalize
signal = centralize(signal)
noisy_signal = centralize(noisy_signal)
#filtered_signal = centralize(filtered_signal)
filtered_signal = filtered_signal - 330



# View 1
plt.plot(signal, 'black')
plt.plot(noisy_signal, 'red')
plt.plot(filtered_signal, 'green')
plt.xlim(212800, 214200)
plt.ylim(-100, 500)
plt.show()

"""
# View 2
plt.xlim(335000, 365000)
plt.show()

# View 3
plt.xlim(435000, 460000)
plt.show()

# View 4
plt.xlim(1.095e6, 1.120e6)
plt.show()

# View 5
plt.xlim(1.190e6, 1.215e6)
plt.show()

# View 6
plt.xlim(1.485e6, 1.510e6)
plt.show()
"""


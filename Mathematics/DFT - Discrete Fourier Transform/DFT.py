#############################
#  Author: Peizhi Yan       #
#############################

import numpy as np

def DFT_1d(x):
    """
    Compute the discrete Fourier Transform of the 1D array x
        x - input signal
    """
    T = x.shape[0]
    t = np.arange(T)
    k = t.reshape((T, 1))
    e = np.exp(-2j * np.pi * k * t / T) # 'j' is the unit imaginary value in python
    return np.dot(e, x)/T

def DFT_2d(x):
    """
    Compute the discrete Fourier Transform of the 2D matrix x
        x - input 2D signal: shape [N, M]
    """
    N, M = x.shape
    n = np.arange(N)
    m = np.arange(M)
    l = n.reshape((N, 1))
    k = m.reshape((M, 1))
    z = np.zeros([N, M], dtype=np.complex64) # to store the result
    # Because DFT is separable, we can do first do it row-wise, then column-wise
    e = np.exp(-2j * np.pi * k * m / M)
    for i in range(N):
        z[i, :] = np.dot(e, x[i]) # equivalent to DFT_1d(x[i, :])
    e = np.exp(-2j * np.pi * l * n / N)
    for j in range(M):
        z[:, j] = np.dot(e, z[:, j]) # equivalent to DFT_1d(z[:, j])
    return z/(M*N)


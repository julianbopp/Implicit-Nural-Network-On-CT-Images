import numpy as np


def SNR(x, xhat):
    diff = x - xhat
    return -20 * np.log10(np.linalg.norm(diff) / np.linalg.norm(x))


def addGaussianNoise(signal, mean, std):
    return signal + np.random.normal(mean, std, signal.shape)

import numpy as np


def tanh(Inp, derivative=False):
    if derivative:
        return 1-((tanh(Inp)) ** 2)
    return (np.exp(Inp) - np.exp(-1*Inp))/(np.exp(Inp) + np.exp(-1*Inp))
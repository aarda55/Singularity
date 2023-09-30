import numpy as np


def Softplus(Inp, derivative=False):
    if derivative:
        return (1 /(1 + np.exp(-Inp)))
    return (np.log(1 + np.exp(Inp)))
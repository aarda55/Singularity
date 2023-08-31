import numpy as np


def Leaky_ReLu(Inp, derivative=False):
    if derivative:
        return np.where(Inp>0, 1, 0.01) 
    return np.maximum(0.01*Inp, Inp)
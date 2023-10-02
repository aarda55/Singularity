import numpy as np


def ReLU(Inp, derivative=False):
    if derivative:
        return Inp > 0
    return np.maximum(Inp, 0)
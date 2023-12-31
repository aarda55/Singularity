import numpy as np


def ELU(Inp, derivative=False):
    if derivative:
        if Inp > 0:
            return 1
        else:
            return np.exp(Inp)
    if Inp > 0:
        return Inp
    else:
        return (np.exp(Inp)-1)
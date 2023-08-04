import numpy as np


def ELU(Inp):
    if Inp > 0:
        return Inp
    else:
        return (np.exp(Inp)-1)

def derivELU(Inp):
    if Inp > 0:
        return 1
    else:
        return np.exp(Inp)
import numpy as np


def tanh(Inp):
    return (np.exp(Inp) - np.exp(-1*Inp))/(np.exp(Inp) + np.exp(-1*Inp))

def derivtanh(Inp):
    return 1-((tanh(Inp)) ** 2)
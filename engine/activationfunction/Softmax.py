import numpy as np


def Softmax(Inp):
    return np.exp(Inp) / sum(np.exp(Inp))
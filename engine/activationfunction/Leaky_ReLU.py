import numpy as np


def Leaky_ReLu(Inp):
    return np.maximum(0.01*Inp, Inp)

def derivLeaky_ReLu(Inp):
    return np.where(Inp>0, 1, 0.01) 
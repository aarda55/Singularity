import numpy as np


def ReLU(Inp):
    return np.maximum(Inp, 0)

def derivRelU(Inp):
    return Inp > 0
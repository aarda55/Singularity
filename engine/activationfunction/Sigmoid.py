import numpy as np


def Sigmoid(Inp, derivative=False):
    if derivative:
        return (Sigmoid(Inp)*(1-(Sigmoid(Inp))))
    return 1 / (1 + np.exp(-1*(Inp)))
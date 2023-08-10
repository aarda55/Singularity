import numpy as np
from Sigmoid import Sigmoid


def Swish1(Inp, derivative=False):
    if derivative:
        return Sigmoid(Inp) * (1 + Inp * (1 - Sigmoid(Inp)))
    return (Inp*Sigmoid(Inp))
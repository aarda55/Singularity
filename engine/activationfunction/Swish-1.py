import numpy as np
from Sigmoid import Sigmoid


def Swish1(Inp):
    return (Inp*Sigmoid(Inp))

def derivSwish1(Inp):
    return Sigmoid(Inp) * (1 + Inp * (1 - Sigmoid(Inp)))
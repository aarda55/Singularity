import numpy as np


def Sigmoid(Inp):
    return 1 / (1 + np.exp(-1*(Inp)))

def derivSigmoid(Inp):
    return (Sigmoid(Inp)*(1-(Sigmoid(Inp))))
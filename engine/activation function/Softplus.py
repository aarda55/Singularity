import numpy as np


def Softplus(Inp):
    return (np.log(1 + np.exp(Inp)))

def derivSoftplus(Inp):
    return (1 /(1 + np.exp(-Inp)))
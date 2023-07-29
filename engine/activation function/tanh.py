import numpy as np

def tanh(InZ):
    return (np.exp(InZ) - np.exp(-1*InZ))/(np.exp(InZ) + np.exp(-1*InZ))
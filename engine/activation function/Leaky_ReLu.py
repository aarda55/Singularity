import numpy as np

def Leaky_ReLu(InZ):
    return np.maximum(0.01*InZ, InZ)
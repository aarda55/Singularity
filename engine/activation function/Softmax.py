import numpy as np


def Softmax(InZ):
    return np.exp(InZ) / sum(np.exp(InZ))
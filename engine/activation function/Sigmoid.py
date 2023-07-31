import numpy as np


def Sigmoid(InZ):
    return 1 / 1 + np.exp(-1*(InZ))
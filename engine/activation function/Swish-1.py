import numpy as np
from Sigmoid import Sigmoid


def Swish1(InZ):
    return (InZ*Sigmoid(InZ))
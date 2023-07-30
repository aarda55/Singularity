import numpy as np

def ELU(InZ):
    if InZ > 0:
        return InZ
    else:
        return (np.exp(InZ)-1)
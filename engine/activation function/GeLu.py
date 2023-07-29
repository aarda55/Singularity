import numpy as np
import tanh

def GeLu(InZ):
    return (1+ tanh(np.sqrt(2/np.pi)*(InZ + 0.044715*(InZ*InZ*InZ))))
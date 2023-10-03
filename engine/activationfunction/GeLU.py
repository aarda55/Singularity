import numpy as np
from tanh import tanh


def GeLU(Inp, derivative=False):
    if derivative:
        return 0.5 + 0.5 * Inp * (0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0))))
    return (1+tanh(np.sqrt(2/np.pi)*(Inp + 0.044715*(Inp ** 3))))
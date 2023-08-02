import numpy as np

def hyperInit(numoflayers, Inshape, Outshape):
    for i in range (0, numoflayers):
        globals()['W%s' % i] = np.random.randn(1,2)
        globals()['b%s' % i] = np.random.randn(2,2)
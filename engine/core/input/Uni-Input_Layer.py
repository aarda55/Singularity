import numpy as np
from weightbiasgen import hyperInit

def Input_layer(Input_data, numoflayers, Outshape):
    if Input_data == np.array or isinstance(Input_data, list) == True:
        Input_data = np.array(Input_data)
        Input_shape = Input_data.shape
        hyperInit(numoflayers, Input_shape, Outshape)
        return Input_data
    else:
        print("Singularity: Object being passed has to either be a python list or a numpy array!")
import numpy as np


def Input_layer(Input_data):
    if Input_data == np.array or isinstance(Input_data, list) == True:
        Input_data = np.array(Input_data)
        return Input_data
    else:
        print("Singularity: Object being passed has to either be a python list or a numpy array!")
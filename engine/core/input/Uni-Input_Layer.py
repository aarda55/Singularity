import numpy as np

def Input_layer(X_dimension, Y_dimension, Input_data):
    array_shape = (X_dimension, Y_dimension)
    arr_in = np.empty(array_shape)
    if Input_data == np.array:
        pass
    if isinstance(Input_data, list) == True:
        Input_data = np.array(Input_data)
    else:
        print("Singularity: Object being passed has to either be a python list or a numpy array!")
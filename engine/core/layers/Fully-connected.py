import numpy as np


class layers:
    def __init__(self):
        self.layers = []

    def add_layer(self, neurons, activation_function):
        self.layers.append({
            'neurons': neurons,
            'activation_function': activation_function,
            'weights': None,
            'biases': None
        })

    def initialize_parameters(self, input_size, output_size):
        prev_layer_size = input_size
        for layer in self.layers:
            layer['weights'] = np.random.randn(prev_layer_size, layer['neurons'])
            layer['biases'] = np.zeros((1, layer['neurons']))
            prev_layer_size = layer['neurons']

        self.layers[-1]['weights'] = np.random.randn(prev_layer_size, output_size)
        self.layers[-1]['biases'] = np.zeros((1, output_size))

    def Input_layer(Input_data, Input_shape, Out_shape):
        initialize_parameters(Input_shape, Out_shape)
        if Input_data == np.array or isinstance(Input_data, list) == True:
            Input_data = np.array(Input_data)
            Input_shape = Input_data.shape
            return Input_data, Input_shape, Out_shape
        else:
            print("Singularity: Object being passed has to either be a python list or a numpy array!")
    
    def forward_prop(self, X):
        input_data = X
        for layer in self.layers:
            layer_activation = np.dot(input_data, layer['weights']) + layer['biases']
            layer_output = layer['activation_function'](layer_activation)
            Output_data = layer_output

        return Output_data

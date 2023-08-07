import numpy as np
from ...activationfunction import ReLU

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
    
    def Evaluate(self, X):
        input_data = X
        if X.shape == Input_shape: return print("Singularity: Input shape of Test data has to be the same as Input shape of training data. Instead got", X.shape)
        for layer in self.layers:
            layer_activation = np.dot(input_data, layer['weights']) + layer['biases']
            layer_output = layer['activation_function'](layer_activation)
            Output_data = layer_output

        return Output_data

    def train(self, X_train, y_train, learning_rate, epochs, batch_size):
        if learning_rate == none: learning_rate = 0.01
        m = X_train.shape[0] 

        for epoch in range(epochs):
            for i in range(0, m, batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                # Forward-propagation
                layer_outputs = [batch_X]
                for layer in self.layers:
                    layer_activation = np.dot(layer_outputs[-1], layer['weights']) + layer['biases']
                    if layer['activation_function']:
                        layer_output = layer['activation_function'](layer_activation)
                    else:
                        layer_output = layer_activation
                    layer_outputs.append(layer_output)
                predictions = layer_outputs[-1]

                # Backpropagation
                d_predictions = -(np.divide(batch_y, predictions) - np.divide(1 - batch_y, 1 - predictions)) / batch_size

                for i in reversed(range(len(self.layers))):
                    layer = self.layers[i]
                    d_activation = d_predictions
                    if layer['activation_function']:
                        d_activation *= layer['activation_function'](layer_outputs[i+1], derivative=True)

                    d_weights = np.dot(layer_outputs[i].T, d_activation)
                    d_biases = np.sum(d_activation, axis=0, keepdims=True)

                    d_predictions = np.dot(d_activation, layer['weights'].T)

                    # Updated parameters
                    layer['weights'] -= learning_rate * d_weights
                    layer['biases'] -= learning_rate * d_biases
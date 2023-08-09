import numpy as np

class model:
    def __init__(self):
        self.layers = []

    def _one_hot_test(y_tests):
        for y_test in y_tests:
            if np.sum(y_test) != 1:
                return False
            if np.any(y_test != 0) and np.all(y_test != 1):
                return False
        return True

    def add_layer(self, neurons, activation_function):
        self.layers.append({
            'neurons': neurons,
            'activation_function': activation_function,
            'weights': None,
            'biases': None
        })

    def _initialize_parameters(self, input_size, output_size):
        prev_layer_size = input_size

        # Creates random the weights and biases
        for layer in self.layers:
            layer['weights'] = np.random.randn(prev_layer_size, layer['neurons'])
            layer['biases'] = np.zeros((1, layer['neurons']))
            prev_layer_size = layer['neurons']

        self.layers[-1]['weights'] = np.random.randn(prev_layer_size, output_size)
        self.layers[-1]['biases'] = np.zeros((1, output_size))

    def Input_layer(Input_data, Input_shape, Out_shape):

        # Parameters are randomly initialized to start gradient descent
        initialize_parameters(Input_shape, Out_shape)
        if Input_data == np.array or isinstance(Input_data, list) == True:
            Input_data = np.array(Input_data)
            Input_shape = Input_data.shape
            return Input_data, Input_shape, Out_shape
        else:
            print("Singularity: Typeerror: Object being passed has to either be a python list or a numpy array!")
    
    def _Prop(self, X):
        input_data = X

        # Shape has to stay the same else a Valueerror is issued
        if X.shape != Input_shape: return print("Singularity: Valueerror: Input shape of Test data has to be the same as Input shape of training data. Instead got", X.shape)
        
        # Goes through every layer and does thee forward propagation
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

                # Forward-propagation with at first random weights, then optimized weights
                layer_outputs = [batch_X]
                for layer in self.layers:
                    layer_activation = np.dot(layer_outputs[-1], layer['weights']) + layer['biases']
                    if layer['activation_function']:
                        layer_output = layer['activation_function'](layer_activation)
                    else:
                        layer_output = layer_activation
                    layer_outputs.append(layer_output)
                predictions = layer_outputs[-1]

                # Back-propagation for optimizing weights and biases
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
    
    def _predict(self,  X_test):

        # creates a probability distribution for predictions
        propability_dis = self.prop(X_test)
        return np.argmax(propability_dis, axix=1)

    """
    Evaluates the already trained network
    """
    def Evaluate(self, X_test, y_test):

        if(one_hot_test(y_test) == False):
            return print("Singularity: Valueerror: y_test needs to be one-hot-encoded for usage of this function!")

        predictions = self.predict(X_test)
        true_labels = np.argmax(y_test, axis=1)

        correct_predictions = np.sum(predictions == true_labels)
        total_samples = X_test.shape[0]

        accuracy = correct_predictions / total_samples
        return print(f"Singularity: Model-accuracy: {accuracy:.3%}")

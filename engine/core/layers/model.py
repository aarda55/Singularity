import numpy as np
import pickle

class model:
    def __init__(self):
        self.layers = []

    def one_hot_test(y_tests):
        for y_test in y_tests:
            if np.sum(y_test) != 1:
                return False
            if np.any(y_test != 0) and np.all(y_test != 1):
                return False
        return True

    def add_layer(self, neurons, activation_function=None):
        if activation_function is None: activation_function = ReLU
        self.layers.append({
            'neurons': neurons,
            'activation_function': activation_function,
            'weights': None,
            'biases': None
        })

    def initialize_parameters(self, input_size, output_size):
        prev_l_size = input_size

        # Creates random the weights and biases
        for layer in self.layers:
            layer['weights'] = np.random.randn(prev_l_size, layer['neurons'])
            layer['biases'] = np.zeros((1, layer['neurons']))
            prev_l_size = layer['neurons']

        self.layers[-1]['weights'] = np.random.randn(prev_l_size, output_size)
        self.layers[-1]['biases'] = np.zeros((1, output_size))

    def Input_layer(self, Input_data, Input_shape, Out_shape):

        # Input values are checked if they are an array or a list
        if isinstance(Input_data, np.ndarray) or (isinstance(Input_data, list) and len(Input_data) > 0):
            Input_data = np.array(Input_data)
            Input_shape = Input_data.shape
            return Input_data, Input_shape, Out_shape
        else:
            print("Singularity: Type-error: Object being passed has to either be a python list or a numpy array!")
    
    def _Prop(self, X):
        input_data = X

        # Shape has to stay the same else a Value-error is issued
        if X.shape != Input_shape: return print("Singularity: Value-error: Input shape of Test data has to be the same as Input shape of training data. Instead got", X.shape)
        
        # Goes through every layer and does the forward propagation
        for layer in self.layers:
            layer_activation = np.dot(input_data, layer['weights']) + layer['biases']
            layer_output = layer['activation_function'](layer_activation)
            Output_data = layer_output

        return Output_data

    def train(self, X_train, y_train, epochs, batch_size, learning_rate=None):
        if learning_rate is None: learning_rate = 0.01
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
            print(f"Singularity: Training:{epoch+1}/{epochs} - {(epoch+1)/epochs*100}%")


    def predict(self, X, Argmax=None):

        probability_dis_full = self._Prop(X)
        if(Argmax==True):
            return np.argmax(probability_dis_full, axis=1)
        return probability_dis_full

    """
    saves the model data
    """
    def savem(self, path):
        model_data = {
            'layers': []
        }

        for layer in self.layers:
            layer_data = {
                'neurons': layer['neurons'],
                'activation_function': layer['activation_function'],
                'weights': layer['weights'],
                'biases': layer['biases']
            }
            model_data['layers'].append(layer_data)

        with open(path, 'wb') as file:
            pickle.dump(model_data, file)

    """
    Evaluates the already trained network
    """
    def Evaluate(self, X_test, y_test):

        if(self.one_hot_test(y_test) == False):
            return print("Singularity: Value-error: y_test needs to be one-hot-encoded for usage of this function!")

        predictions = self.predict(X_test, True)
        true_labels = np.argmax(y_test, axis=1)

        correct_predictions = np.sum(predictions == true_labels)
        total_samples = X_test.shape[0]

        accuracy = correct_predictions / total_samples
        return print(f"Singularity: Model-accuracy: {accuracy:.3%}")

    @classmethod
    def load_model(cls, path):
        with open(path, 'rb') as file:
            model_data = pickle.load(file)
        loaded_model = cls()

        for layer_data in model_data['layers']:
            neurons = layer_data['neurons']
            activation_function = layer_data['activation_function']
            loaded_model.add_layer(neurons, activation_function)

        loaded_model.initialize_parameters(input_size, output_size)
        
        for i, layer in enumerate(loaded_model.layers):
            layer['weights'] = model_data['layers'][i]['weights']
            layer['biases'] = model_data['layers'][i]['biases']

        return loaded_model
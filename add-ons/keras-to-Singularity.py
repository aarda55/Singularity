from engine.core.layers.model import Model
from tensorflow import keras

def keras_to_Singularity(keras_model):
    
    sing_model = Model()

    for keras_layer in keras_model.layers:
        neurons = keras_layer.units
        activation_function = keras_layer.activation.__name__ if keras_layer.activation else None

        sing_model.add_layer(neurons, activation_function)
    
    sing_model.initialize_parameters(input_size=keras_model.input_shape[1], output_size=keras_model.output_shape[1])
    
    for i, keras_layer in enumerate(keras_model.layers):
        sing_model.layers[i]['weights'] = keras_layer.get_weights()[0]
        sing_model.layers[i]['biases'] = keras_layer.get_weights()[1]

    return sing_model

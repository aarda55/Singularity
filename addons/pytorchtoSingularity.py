import numpy as np
import pickle
import torch
import torch.nn as nn
from engine.core.layers.Model import Model

def convert_pytorch_to_custom_model(pytorch_model):

    if(torch not in sys.modules):
        return ("Package-error: Please install and import pytorch")
    
    Sing_model = Model()
    
    input_size = None
    output_size = None
    
    for name, module in pytorch_model.named_children():
        if isinstance(module, nn.Linear):
            neurons = module.out_features
            activation_function = module.__class__.__name__
            
            if input_size is None:
                input_size = module.in_features 
            Sing_model.add_layer(neurons, activation_function)
    
    if input_size is None or output_size is None:
        raise ValueError("Singularity: Value-Error: Input/output size could not be determined from the PyTorch model.")

    custom_model.initialize_parameters(input_size=input_size, output_size=output_size)
    
    for i, (name, module) in enumerate(pytorch_model.named_children()):
        if isinstance(module, nn.Linear):
            Sing_model.layers[i]['weights'] = module.weight.detach().numpy()
            Sing_model.layers[i]['biases'] = module.bias.detach().numpy()

    return Sing_model

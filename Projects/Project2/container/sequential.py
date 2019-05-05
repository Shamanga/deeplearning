import torch

class Sequential:
    
    def __init__(self, layers):
        
        self.layers = layers
        
    def forward(self, initial_input):
        
        # initial input
        output_single_layer = initial_input
        
        # Iterate through the layers of the network
        # Pass the input to the forward function of the first layer
        # and keep iterating over the layers and passing the output
        # of the layer before to the one after
        for layer in self.layers:
            output_single_layer = layer.forward(output_single_layer)
        
        # The last output of the network
        return output_single_layer
    
    def backward(self, initial_backward_input):
        
        # Starting with the derivative of the loss
        # We backpropagate by calling the backward
        # function of each layer 
        output_single_layer_backward = initial_backward_input
        
        for layer in self.layers[::-1]:
            
            output_single_layer_backward = layer.backward(output_single_layer_backward)
            
        return output_single_layer_backward
    
    def param(self):
        
        parameters_of_each_layer = []
        for layer in self.layers:
            
            parameters_of_each_layer.append(layer.param)
            
        return parameters_of_each_layer
            
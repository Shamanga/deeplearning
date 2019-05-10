import torch

class Sequential:
    
    """
    Implementation of the Sequential layer
    Functions:
        - forward: Forward pass
        - backward: backward pass
        - param: Parameters of each layer
    
    """
    
    def __init__(self, layers):
        """
        layers: The layers of the network
        """
        
        self.layers = layers
        
    def forward(self, initial_input):
        
        """
        The forward pass
        
        Args: 
            - initial_input: The initial input that starts at the first layer
        Returns:
            - output_single_layer: The output of the last layer
        """
        
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
        
        """
        The Backward pass
        
        Args: 
            - initial_backward_input: The gradient of the loss, as the first input to the backpass
        Returns:
            - output_single_layer_backward: The output of the derivatives in the end
        """
        
        # Starting with the derivative of the loss
        # We backpropagate by calling the backward
        # function of each layer 
        output_single_layer_backward = initial_backward_input
        
        for layer in self.layers[::-1]:
            
            output_single_layer_backward = layer.backward(output_single_layer_backward)
            
        return output_single_layer_backward
    
    def param(self):
        
        """
        The Parameters
        
        Args: 

        Returns:
            - parameters_of_each_layer: The parameters of each layer
        """
        
        parameters_of_each_layer = []
        for layer in self.layers:
            
            parameters_of_each_layer.append(layer.param())
            
        return parameters_of_each_layer
            
            
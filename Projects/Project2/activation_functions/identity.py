import torch
import sys
sys.path.append("..")
from module import Module

class Identity(Module):
    
    """
    Implementation of the Identity module
    Functions:
        - forward: Forward pass
        - backward: backward pass
        - _identity: Returns the same input with no changes
        - _identity_p: Derivative of Identity. Gradient equals to one.
    
    """
    
    def forward(self, *input):
        """
        The forward pass
        
        Args: 
            - input: The input from the previous layer
        Returns:
            - forward_pass: The output of the activation function
        """
        # Grabbing the input, which is the output of the previous layer
        self.input_from_layer = input[0]
        forward_pass = self.identity(self.input_from_layer)
        
        return forward_pass
    
    
    def backward(self, *gradwrtoutput):
        """
        The Backward pass
        
        Args: 
            - gradwrtoutput: The gradient from the layer ahead 
        Returns:
            - backward_pass: The derivative of the input multiplied by the gradient of the layer ahead
        """
        
        # Calculating the derivative of the input to the Identity from the previous layer
        derivatives = self.identity_p(self.input_from_layer)
        backward_pass = derivatives * gradwrtoutput[0]
        
        return backward_pass
    
    
    def _identity(self, to_compute):
        """
        Identity function which essentially returns back the same input
        Args:
            - to_compute: A tensor that we would like to apply the activation function on
        returns:
            - to_compute: A tensor with activation function applied to it
        """
        return to_compute
    
    def _identity_p(self, to_compute):
        """
        The identity activation function derivative
        Args:
            - to_compute: A tensor that we would like to apply the derivative of the activation function
        returns:
            - to_compute: A tensor with the derivatives replacing the original values
        """
        return torch.ones_like(to_compute)

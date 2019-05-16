import torch
import sys
sys.path.append("..")
from module import Module

class Negative(Module):
    """
    Implementation of the Negative module
    Functions:
        - forward: Forward pass
        - backward: backward pass
        - _negative: Negative activation function that reutnrs the negative of the input
        - _negative_p: Derivative of negative. Gradient equals to negative one
    
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
        forward_pass = self._negative(self.input_from_layer)
        
        return forward_pass
    
    
    def backward(self, *gradwrtoutput):
        """
        The Backward pass
        
        Args: 
            - gradwrtoutput: The gradient from the layer ahead 
        Returns:
            - backward_pass: The derivative of the input multiplied by the gradient of the layer ahead
        """
        
        # Calculating the derivative of the input to the Negative  from the previous layer
        derivatives = self._negative_p(self.input_from_layer)
        backward_pass = derivatives * gradwrtoutput[0]
        
        return backward_pass
    
    
    def _negative(self, to_compute):
        """
        Negative function which essentially returns back the negative of the input
        Args:
            - to_compute: A tensor that we would like to apply the activation function on
        returns:
            - to_compute: A tensor with activation function applied to it
        """
        return -to_compute
    
    def _negative_p(self, to_compute):
        """
        The negative activation function derivative
        Args:
            - to_compute: A tensor that we would like to apply the derivative of the activation function
        returns:
            - to_compute: A tensor with the derivatives replacing the original values
        """
        return -torch.ones_like(to_compute)
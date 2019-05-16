import torch
import sys
sys.path.append("..")
from module import Module

class Sigmoid(Module):
    """
    Implementation of the Sigmoid module
    Functions:
        - forward: Forward pass
        - backward: backward pass
        - _sigmoid: Applying the sigmoid function to the input
        - _sigmoid_p: Derivative of the sigmoid.
    
    """
    # Forward pass
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
        # Passing the input through the activation function
        forward_pass = self._sigmoid(self.input_from_layer)
        
        return forward_pass
    
    # Backward pass
    def backward(self, *gradwrtoutput):
        
        """
        The Backward pass
        
        Args: 
            - gradwrtoutput: The gradient from the layer ahead 
        Returns:
            - backward_pass: The derivative of the input multiplied by the gradient of the layer ahead
        """
        
        # Calculating the derivative of the input to the Sigmoid from the previous layer
        derivatives = self._sigmoid_p(self.input_from_layer)
        backward_pass = derivatives * gradwrtoutput[0]
        
        return backward_pass
    
    def _sigmoid(self, to_compute):
        """
        Sigmoid function
        Args:
            - to_compute: A tensor that we would like to apply the activation function on
        returns:
            - output_to_compute: A tensor with activation function applied to it
        """
        output_to_compute = 1./ (1. + torch.exp(-to_compute))
        return output_to_compute
    
    def _sigmoid_p(self, to_compute):
        """
        The Sigmoid activation function derivative
        Args:
            - to_compute: A tensor that we would like to apply the derivative of the activation function
        returns:
            - output_to_compute: A tensor with the derivatives replacing the original values
        """
        output_to_compute = self._sigmoid(to_compute) * (1 - self._sigmoid(to_compute))
        #output_to_compute = torch.exp(-to_compute) / (1 + torch.exp(-to_compute)) ** 2
        return output_to_compute

import torch
import sys
sys.path.append("..")
from module import Module

class Tanh(Module):
    """
    Implementation of the Relu module
    Functions:
        - forward: Forward pass
        - backward: backward pass
        - _tanh: Tanh function implementation
        - _tanh_p: Derivative of tanh.
    
    
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
        forward_pass = self._tanh(self.input_from_layer)
        
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
        
        # Calculating the derivative of the input to the Tanh from the previous layer
        derivatives = self._tanh_p(self.input_from_layer)
        backward_pass = derivatives * gradwrtoutput[0]
        
        return backward_pass
    
    # Tanh activation function
    def _tanh(self, to_compute):
        
        """
        The Tanh activation function
        
        Args:
            - to_compute: A tensor that we would like to apply the activation function on
        Returns:
            - activation_output: Tensor applied to it the activation function
        """
        
        numerator = torch.exp(to_compute) - torch.exp(-to_compute)
        denominator = torch.exp(to_compute) + torch.exp(-to_compute)
        
        activation_output = numerator/denominator
        
        return activation_output
        
    # Tanh derivative
    def _tanh_p(self, to_compute):
        
        """
        The tanh activation function derivative
        
        Args:
            - to_compute: A tensor that we would like to apply the derivative of the activation function
        returns:
            - activation_derivative: A tensor with the derivatives replacing the original values
        """
        activation_derivative = (1 - torch.pow(self._tanh(to_compute),2))
        
        return activation_derivative

import torch
import sys
sys.path.append("..")
from module import Module

class Relu(Module):
    """
    Implementation of the Relu module
    Functions:
        - forward: Forward pass
        - backward: backward pass
        - _relu: Rectified linear unit, passing anything that is positive and suppressing
                negative number to zero
        - _relu_p: Derivative of Relu. Gradient equals to zero if positive number, otherwise zero
    
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
        forward_pass = self._relu(self.input_from_layer)
        
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
        
        # Calculating the derivative of the input to the Relu from the previous layer
        derivatives = self._relu_p(self.input_from_layer)
        backward_pass = derivatives * gradwrtoutput[0]
        return backward_pass
    
    # Relu activation function
    def _relu(self, to_compute):
        
        """
        The Relu activation function
        
        Args:
            - to_compute: A tensor that we would like to apply the activation function on
        Returns:
            - activation_output: Tensor applied to it the activation function
        """
        activation_output = torch.clamp(to_compute, min =0)
        return activation_output
    
    # Relu derivative
    def _relu_p(self, to_compute):
        """
        The Relu activation function derivative
        
        Args:
            - to_compute: A tensor that we would like to apply the derivative of the activation function
        returns:
            - to_compute: A tensor with the derivatives replacing the original values
        """

        to_compute[to_compute>0] = 1
        to_compute[to_compute<=0] = 0

        return to_compute
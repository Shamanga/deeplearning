import torch
import sys
sys.path.append("..")
from module import Module

class Softmax(Module):
    
    """
    Implementation of the Softmax module
    Functions:
        - forward: Forward pass
        - backward: backward pass
        - _softmax: Softmax function implementation
        - _softmax_p: Derivative of Softmax.
    
    """
    
    def forward(self, *input):
        
        """
        The forward pass
        
        Args: 
            - input: The input from the previous layer
        Returns:
            - forward_pass: The output of the function
        """
        # Grabbing the input, which is the output of the previous layer
        self.input_from_layer = input[0]
        # Passing the input through the activation function
        forward_pass = self._softmax(input_from_layer)
        
        return forward_pass
    
    def backward(self, *gradwrtoutput):
        
        """
        The Backward pass
        
        Args: 
            - gradwrtoutput: The gradient from the layer ahead 
        Returns:
            - backward_pass: The derivative of the input multiplied by the gradient of the layer ahead
        """
        # Calculating the derivative of the input to the Softmax from the previous layer
        derivatives = self._softmax_p(self.input_from_layer)
        backward_pass = derivatives * gradwrtoutput[0]
        
        return backward_pass
        
    
    def _softmax(self, to_compute):
        
        """
        The Tanh activation function
        
        Args:
            - to_compute: A tensor that we would like to apply the activation function on
        Returns:
            - activation_output: Tensor applied to it the activation function
        """
        
        input_to_compute_v = input_to_compute.view(-1,1)
        
        norm_value = input_to_compute_v.max()
        
        stable_input_to_compute_v = input_to_compute_v - norm_value
        
        exponentials = torch.exp(stable_input_to_compute_v)
        
        sum_exponentials = torch.sum(exponentials)
        
        activation_output = (exponentials/sum_exponentials).view(-1)
        
        return activation_output
    
    def _softmax_p(self, to_compute):
        
        """
        The tanh activation function derivative
        
        Args:
            - to_compute: A tensor that we would like to apply the derivative of the activation function
        returns:
            - activation_derivative: A tensor with the derivatives replacing the original values
        """
        
        softmax_res = self._softmax(to_compute)
        
        diag_softm = torch.diag(softmax_res)
        
        activation_derivative = torch.FloatTensor(diag_softm.shape[0], diag_softm.shape[0])
        
        for i in range((diag_softm.shape[0])):
            for j in range((diag_softm.shape[0])):
                if i == j:
                    activation_derivative[i][j] = softmax_res[i] * (1 - softmax_res[i])
                else:
                    activation_derivative[i][j] = -softmax_res[i] * softmax_res[j]
                    
        return activation_derivative
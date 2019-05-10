import torch
import sys
sys.path.append("..")
from module import Module


class Linear(Module):
    
    """
    Implementation of the Linear Module
    Functions:
        - forward: The forward pass
        - backward: The backward pass
        - param: The parameters of the layer
    
    """
    
    def __init__(self, in_features, out_features):
        """
        in_features: The number of input features
        out_features: The number of output features
        """
        # Number of input neurons
        self.in_features = in_features
        # Number of output neurons
        self.out_features = out_features
        
        # Initializing the weights with Xavier’s initialization
        # First generate the weights from a normal distribution with mean 0 and std 1
        # Then multiply the samples by sqrt(1 / (number_of_input_neurons + number_of_output_neurons))
        self.weights = torch.mul(torch.Tensor(out_features, in_features).normal_(mean=0, std=1), \
                                torch.sqrt(torch.FloatTensor([1/ (self.in_features + self.out_features)])))
        
        # Zero bias initialization
        self.bias = torch.Tensor(out_features).zero_()
        
        # Initiliazing the derivative tensor
        self.derivative_layer_weights = torch.Tensor(self.weights.size()).zero_()
        self.derivative_layer_bias = torch.Tensor(self.bias.size()).zero_()
    
        
    def forward(self, *input):
        
        """
        The forward pass
        
        Args: 
            - input: The input from the previous layer
        Returns:
            - forward_pass: The output of the function
        """
        
        # Input from the layer
        self.input_from_layer = input[0]
        # Calculating the output, which is basically the multiplication
        # of the weights with the input layer and adding the bias
        self.forward_pass = torch.mv(self.weights, self.input_from_layer) + self.bias
        
        return self.forward_pass

    def backward(self, *gradwrtoutput):
        
        """
        The Backward pass
        
        Args: 
            - gradwrtoutput: The gradient from the layer ahead 
        Returns:
            - backward_pass: The derivative of the input multiplied by the gradient of the layer ahead
        """
        
        # Computing the gradient by multiplying by the gradient of the previous layer
        weights_transpose = self.weights.t()
        backward_pass = weights_transpose.mv(gradwrtoutput[0])

        # Compute the weights gradient by multiplying the gradient by the input 
        derivative_weights = gradwrtoutput[0].view(-1, 1).mm(self.input_from_layer.view(1, -1))
        self.derivative_layer_weights.add_(derivative_weights)

        # Compute the gradient of the bias
        self.derivative_layer_bias.add_(gradwrtoutput[0])
        
        return backward_pass

    def param(self):
        
        """
        The Parameters
        
        Args: 

        Returns:
            - linear_layer_parameters: The parameters of the linear layer
        """
        linear_layer_parameters = [self.weights, self.derivative_layer_weights,\
                self.bias, self.derivative_layer_bias]
        return linear_layer_parameters
    
    
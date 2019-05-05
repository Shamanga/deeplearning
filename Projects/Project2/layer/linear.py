import torch
from ..module import Module


class Linear(Module):
    
    def __init__(self, in_features, out_features):
        
        # Number of input neurons
        self.in_features = in_features
        # Number of output neurons
        self.out_features = out_features
        
        # Initializing the weights with Xavier’s initialization
        # First generate the weights from a normal distribution with mean 0 and std 1
        # Then multiply the samples by sqrt(1 / (number_of_input_neurons + number_of_output_neurons))
        self.weight = torch.mul(torch.Tensor(out_features, in_features).normal_(mean=0, std=1), \ 
                                torch.sqrt(torch.FloatTensor([1/ (self.in_features + self.out_features)])))
        
        # Zero bias initialization
        self.bias = torch.Tensor(out_features).zero_()

        
        
    def forward(self, *input):
        
        # Input from the layer
        self.input_from_layer = input[0]
        
        # Calculating the output, which is basically the multiplication
        # of the weights with the input layer and adding the bias
        self.output = torch.mv(weights, self.input_from_layer) + self.bias
        
        
        return self.output

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []
    
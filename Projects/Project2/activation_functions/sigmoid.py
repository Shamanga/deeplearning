import torch
from ..module import Module

class Sigmoid(Module):
    
    def forward(self, *input):
        
        self.output = input[0]
        
        return self.sigmoid(self.output)
    
    
    def backward(self, *gradwrtoutput):
        
        derivatives = self.sigmoid_p(self.output)
        
        return derivatives * gradwrtoutput[0]
    
    
    def sigmoid(self, x):
        return 1./ (1. + torch.exp(-x))
    
    def sigmoid_p(self, x):
        return torch.exp(x) / (1 + torch.exp(x)) ** 2

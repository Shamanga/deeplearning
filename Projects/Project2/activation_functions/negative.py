import torch
from ..module import Module

class Negative(Module):
    
    def forward(self, *input):
        
        self.output = input[0]
        
        return self.sigmoid(self.output)
    
    
    def backward(self, *gradwrtoutput):
        
        derivatives = self.sigmoid_p(self.output)
        
        return derivatives * gradwrtoutput[0]
    
    
    def negative(self, x):
        return -x
    
    def negative_p(self, x):
        return -torch.ones_like(x)

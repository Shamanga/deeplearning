import torch
from ..module import Module

class Identity(Module):
    
    def forward(self, *input):
        
        self.output = input[0]
        
        return self.identity(self.output)
    
    
    def backward(self, *gradwrtoutput):
        
        derivatives = self.identity_p(self.output)
        
        return derivatives * gradwrtoutput[0]
    
    
    def identity(self, x):
        return x
    
    def identity_p(self, x):
        return torch.ones_like(x)

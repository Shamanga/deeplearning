import torch
from ..module import Module

class Relu(Module):
    
    def forward(self, *input):
        
        self.output = input[0]
        
        return self.relu(self.output)
    
    
    def backward(self, *gradwrtoutput):
        
        derivatives = self.relu_p(self.output)
        
        return derivatives * gradwrtoutput[0]
    
    
    def relu(self, x):
        
        return torch.clamp(x, min =0)
    
    def relu_p(self, x):
        # (torch.sign(x)+1)/2

        x[x>0] = 1
        x[x<=0] = 0

        return x

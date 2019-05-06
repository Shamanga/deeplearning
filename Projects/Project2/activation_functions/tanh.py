import torch
from ..module import Module

class Tanh(Module):
    
    def forward(self, *input):
        
        self.output = input[0]
        
        return self.tanh(self.output)
    
    
    def backward(self, *gradwrtoutput):
        
        derivatives = self.tanh_p(self.output)
        
        return derivatives * gradwrtoutput[0]
    
    def tanh(self, to_compute):
        
        numerator = torch.exp(to_compute) - torch.exp(-to_compute)
        denominator = torch.exp(to_compute) + torch.exp(-to_compute)
        return numerator/denominator
        
        
    def tanh_p(self, x):
        # 4 / (torch.exp(x) + torch.exp(-x)) ** 2
        return (1 - torch.pow(self.tanh(x),2))

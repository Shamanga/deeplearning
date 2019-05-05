import torch

class Mse:
    
    def mse(predicted_output: torch.FloatTensor, target_output: torch.FloatTensor):
        
        # 1/2n (x - y) ^2
        return (torch.pow(predicted_output - target_output, 2).mean()) / 2

    def mse_p(predicted_output: torch.FloatTensor, target_output: torch.FloatTensor):
        
        # 1/n (x - y)
        return (predicted_output - target_output) / len(predicted_output)
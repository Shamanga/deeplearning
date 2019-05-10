import torch

class Mse:
    
    """
    Implementation of the Mean Squared Error
    Functions:
        - mse: Computing the mean squared error
        - mse_p: The derivative of the mean squared error
    
    """
    
    
    def mse(self, predictions, targets):
        """
        Computes mean square error between targets
        and predictions. 
        Args: 
            - predictions: Predictions of the network
            - targets: The true targets of the samples
        Returns: 
            - mse_loss: Mean squared error loss
        """
        # (x - y) ^2
        mse_loss = torch.pow(predictions - targets, 2).sum()
        return mse_loss

    def mse_p(self, predictions, targets):
        """
        Computes mean square derivative between targets
        and predictions. 
        Args: 
            - predictions: Predictions of the network
            - targets: The true targets of the samples
        Returns: 
            - derivative: Derivative of mean squared error loss
        """
        
        # 2 (x - y)
        derivative = 2*(predictions - targets)
        return derivative
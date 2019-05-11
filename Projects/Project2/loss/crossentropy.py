import torch

class CrossEntropy:
    """
    Implementation of the Cross entropy loss
    Functions:
        - cross_entropy: Computing the cross entropy
        - cross_entropy_p: The derivative of the cross entropy
    
    """
    
    def cross_entropy(self, prediction, target, epsilon=1e-20):
        """
        Computes cross entropy between targets
        and predictions. 
        Args: 
            - predictions: Predictions of the network
            - targets: The true targets of the samples
            - epsilon: A value to avoid overflow of numbers
        Returns: 
            - log_loss: The  cross entropy loss
        """
        # Clamping the predictions between epsilon and 1 - epsilon
        predictions_clamped = prediction.clamp(epsilon, 1-epsilon)
        # Obtaining the probabilities of the target class
        to_compute_loss = predictions_clamped[target.max(0)[1]]
        # Computing the loss
        log_loss = -torch.log(to_compute_loss)

        return log_loss
    
    
    def cross_entropy_p(self, prediction, target, epsilon=1e-20):
        
        """
        Computes cross entropy derivative between targets
        and predictions.
        Args: 
            - prediction: Predictions of the network
            - target: The true targets of the samples
            - epsilon: A value to avoid overflow of numbers
        Returns: 
            - derivative: The derivative of the cross entropy
        """

        # Clamping the predictions between epsilon and 1 - epsilon
        prediction_clamped = prediction.clamp(epsilon, 1-epsilon)

        # Computing derivative
        derivative = prediction_clamped - target.type(torch.FloatTensor)

        return derivative
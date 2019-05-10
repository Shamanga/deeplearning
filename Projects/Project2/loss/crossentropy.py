import torch

class CrossEntropy:
    """
    Implementation of the Cross entropy loss
    Functions:
        - cross_entropy: Computing the cross entropy
        - cross_entropy_p: The derivative of the cross entropy
    
    """
    
    def cross_entropy(predictions, targets, epsilon=1e-20):
        """
        Computes cross entropy between targets
        and predictions. 
        Args: 
            - predictions: Predictions of the network
            - targets: The true targets of the samples
            - epsilon: A value to avoid overflow of numbers
        Returns: 
            - average_ce_loss: The average cross entropy loss
        """
        # Clamping the predictions between epsilon and 1 - epsilon
        predictions_clamped = predictions.clamp(epsilon, 1-epsilon)
        # Obtaining the probabilities of the target class
        to_compute_loss = predictions_clamped.gather(1, targets.unsqueeze(1))

        # Computing the loss
        log_loss = -torch.log(to_compute_loss)

        # Computing the mean of the loss
        average_ce_loss = torch.mean(log_loss)

        return average_ce_loss
    
    
    def cross_entropy_p(predictions, targets, epsilon=1e-20):
        
        """
        Computes cross entropy derivative between targets
        and predictions.
        Args: 
            - predictions: Predictions of the network
            - targets: The true targets of the samples
            - epsilon: A value to avoid overflow of numbers
        Returns: 
            - derivative: The derivative of the cross entropy
        """
        
        # Number of examples
        numb_ex = targets.shape[0]

        # Clamping the predictions between epsilon and 1 - epsilon
        predictions_clamped = predictions.clamp(epsilon, 1-epsilon)

        # Computing derivative
    
        predictions_clamped[range(predictions_clamped.shape[0]), targets] -= 1

        derivative = predictions_clamped / numb_ex

        return derivative
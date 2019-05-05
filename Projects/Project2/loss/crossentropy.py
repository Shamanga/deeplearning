import torch

class CrossEntropy:
    
    def cross_entropy_torch(predictions, targets, epsilon=1e-20):
        """
        Computes cross entropy between targets
        and predictions. 
        Input: predictions (N, #ofclasses) FloatTensor
               targets (N, 1) LongTensor        
        Returns: average loss
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
    
    
    def cross_entropy_troch_p(predictions, targets, epsilon=1e-20):
        
        """
        Computes cross entropy derivative between targets
        and predictions. 
        Input: predictions (N, #ofclasses) FloatTensor
               targets (N, 1) LongTensor        
        Returns: derivative
        """
        
        # Number of examples
        numb_ex = targets.shape[0]

        # Clamping the predictions between epsilon and 1 - epsilon
        predictions_clamped = predictions.clamp(epsilon, 1-epsilon)

        # Computing derivative
    
        predictions_clamped[range(predictions_clamped.shape[0]), targets] -= 1

        derivative = predictions_clamped / numb_ex

        return derivative
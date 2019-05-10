import torch

class RunNetwork:

    """
    Implementation of the RunNetwork class
    Functions:
        - run_network: Training and Testing the network

    """

    def __init__(self, epochs, train_class, test_class, model, loss, optimizer, samples):

        """
        epoch: The number of epochs to train the network on
        train_class: The training class that contains the train data
        test_class: The testing class that contains the test data
        model: The model (Seqeuntial + layers)
        loss: The loss function
        optimizer: The optimizer function used
        samples: The number of samples to be training on every epoch
        """
        self.epochs = epochs
        self.train_class = train_class
        self.test_class = test_class
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.samples = samples
    
    
    def run_network(self):
        """
        Running the network over the specified number of epochs
        For each epoch, we train over a full batch and then we
        test to get the 4 parameters that are needed to evaluate 
        the network which are the training accuracy, the training loss,
        the testing loss, and testing accuracy
        """
        for epoch in range(self.epochs):
            
            train_accuracy, train_average_loss = self.network_computations(training=True)
            test_accuracy, test_average_loss = self.network_computations(training=False)
            
                    
            print("Epoch: {}/{} ---- Train Accuracy : {} ---- Train Loss: {} ---- Test Accuracy: {} ---- Test Loss: {} "\
                  .format(epoch+1,self.epochs,train_accuracy, round(train_average_loss.item(),3),test_accuracy,round(test_average_loss.item(),3)))

    def network_computations(self,training = True):
        
        """
        Running the network either in training or testing mode.
        In training mode the parameters are updated and the backward pass is performed
        
        Args:
            training: Whether we want to perform training or not
        Returns:
            accuracy: The accuracy of this epoch
            average_loss: The average loss of this epoch
        """
        # Number of correctly predicted samples
        predicted_correctly = 0
        # To aggregate the loss
        total_loss = 0
        
        # If we are training then we fetch the training data class
        if training:
            data_loader = self.train_class
        # If we are testing then we fetch the testing data class
        else:
            data_loader = self.test_class
        
        # Iterating through the data batches
        for batch_data, batch_target in data_loader.yield_data():
            
            # Iterating over each data point in the batch
            for sample,target in zip(batch_data, batch_target):
                
                # Forward pass of the model
                predicted = self.model.forward(sample)
                # Computing the loss of the predicted value
                single_loss = self.loss.mse(predicted, target)
                
                # If we are training then we need to 
                # compute the derivatives and do the backward pass
                if training:
                    derivative_loss = self.loss.mse_p(predicted, target)
                    self.model.backward(derivative_loss)
                
                # Checking if the predicted value
                # is equivalent to the target value
                if predicted.max(0)[1] == target.max(0)[1]:
                    predicted_correctly += 1
                
                # Accumulating the loss
                total_loss += single_loss
            # If we are training, then we update the parameters
            # Using the optimizer
            if training:
                self.optimizer.update_parameters()
        
        # Calculating the accuracy and average loss
        accuracy = predicted_correctly / self.samples
        average_loss = total_loss / self.samples
        
        return accuracy, average_loss
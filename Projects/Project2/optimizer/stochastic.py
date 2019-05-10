class StochasticGD:
    """
    Implementation of the Stochastic Gradient Descent
    Functions:
        - update_parameters: Updating the wieghts and bias parameters
    
    """
    
    def __init__(self, parameters_list, learning_rate = 0.01):
        """
        parameters_list: A list containing the parameters to be updated
        learning_rate: The learning rate of the algorithm
        """
        self.parameters_list = parameters_list
        self.learning_rate = learning_rate
        
    def update_parameters(self):
        """
        Updating the parameters such as the weights and the bias of the network
        """
        # Iterating through the different parameters
        # The structure of the list can be referred to from the linear layer is
        # [weights, weight_update, bias, bias_update]
        for layer_parameter in self.parameters_list:
            
            try:
                # Fetching the parameters
                weights_ = layer_parameter[0]
                weights_update = layer_parameter[1]
                bias_ = layer_parameter[2]
                bias_update = layer_parameter[3]
                # Updating the parameters
                weights_ -= self.learning_rate*weights_update
                bias_ -= self.learning_rate*bias_update
                # Zeroing the parameters for the next iteration
                bias_update.zero_()
                weights_update.zero_()
            
            # If an exception is raised, means that there are no params
            # function implemented in this layer, hence no parameters back.
            except IndexError:
                continue
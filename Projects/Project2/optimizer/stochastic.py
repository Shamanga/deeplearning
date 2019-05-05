class StochasticGD:
    
    def __init__(self, parameters_list, learning_rate = 0.01):
        
        self.parameters_list = parameters_list
        self.learning_rate = learning_rate
        
    def update_parameters(self):
        
        raise NotImplementedError
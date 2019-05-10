import torch
import math


class DataGenerator:
    
    """
    Implementation of the Data Generator
    Functions:
        - check_target: Checks if a point is within a circle radius
        - generate_data: Generating the data 
        - yield_data: Yielding the data as batches for the network
    
    """
    
    def __init__(self, number_of_examples: int, number_of_features: int, batch_size = 2, shuffle= True):
        """
        number_of_examples: The number of training samples
        number_of_features: The number of features that the samples will hold
        batch_size: The size of the batch to be passed to the network (Default: 2)
        shuffle: Whether we are shuffling the data in the generator or not (Default: True)
        """
        self.number_of_examples = number_of_examples
        self.number_of_features = number_of_features
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Radius of the circle in the exercise problem
        self.radius_circle = 1/math.sqrt(2*math.pi)
        # The center of the circle in the exercise problem
        self.center_circle = [0.5,0.5]
    
    def check_target(self, input_example):
        """
        Checks if the sample is within the radius and assigns a target value
        
        Args:
            input_example: A 2d point 
        Returns:
            1 - If point inside disk
            0 - If point outside disk
        """

        if math.pow(input_example[0] - self.center_circle[0], 2)\
            + math.pow(input_example[1] - self.center_circle[1], 2)\
                < math.pow(self.radius_circle,2):
            return 1
        else:
            return 0


    def generate_data(self):
        """
        Generating the data and assigning the targets to it
        
        Args:
            
        Returns:
            - data: A tensor containing the data
            - one_hot: A tensor containing the targets in a one hot encoding fashion
        """
        # Generating the data
        self.data = torch.FloatTensor(self.number_of_examples, self.number_of_features).uniform_(0,1)
        # Setting up the output tensor
        self.targets = torch.LongTensor(self.number_of_examples)
        # Creating a backup tensor that has indexes as values
        index_targets = torch.arange(0, self.number_of_examples)
        # Iterating through the indexes to fetch the target
        self.targets = index_targets.apply_(lambda i: self.check_target(self.data[i]))
        unsqueezed_targets = self.targets.unsqueeze(1)
        # Transforming the tensor into one hot encoded
        self.one_hot = torch.FloatTensor(self.number_of_examples,2).zero_()
        self.one_hot = self.one_hot.scatter(1, unsqueezed_targets, 1)
        
        return self.data, self.one_hot    
    
    def yield_data(self):
        """
        Yielding the data to the network for training and testing with shuffling or not
        
        Args:
        
        Returns:
            - data: The data with a certain batch size either shuffled or not
            - targets: The targets with a certain batch size either shuffled or not
        
        """
        if self.shuffle==True:
            
            shuffled_indexes = torch.randperm(self.number_of_examples)
            self.data_shuffled = self.data[shuffled_indexes]
            self.targets_shuffled = self.one_hot[shuffled_indexes]
            
            for batch_start in range(0, self.number_of_examples, self.batch_size):
                
                if self.number_of_examples - batch_start >= self.batch_size:
                    yield self.data_shuffled.narrow(0, batch_start, self.batch_size),\
                          self.targets_shuffled.narrow(0, batch_start, self.batch_size)
                else:
                    yield self.data_shuffled.narrow(0, batch_start, self.number_of_examples - batch_start),\
                          self.targets_shuffled.narrow(0, batch_start, self.number_of_examples - batch_start)
        
        else:
            
            for batch_start in range(0, self.number_of_examples, self.batch_size):
                
                if self.number_of_examples - batch_start >= self.batch_size:
                    yield self.data.narrow(0, batch_start, self.batch_size),\
                          self.one_hot.narrow(0, batch_start, self.batch_size)
                else:
                    yield self.data.narrow(0, batch_start, self.number_of_examples - batch_start),\
                          self.one_hot.narrow(0, batch_start, self.number_of_examples - batch_start)
        
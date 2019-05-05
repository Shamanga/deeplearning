import torch
import math


class DataGenerator:
    
    def __init__(self, number_of_examples, number_of_features, batch_size = 2, shuffle= True):
        
        self.number_of_examples = number_of_examples
        self.number_of_features = number_of_features
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.radius_circle = 1/math.sqrt(2*math.pi)
        self.center_circle = [0.5,0.5]
    
    def check_target(self, input_example):


        if math.pow(input_example[0] - self.center_circle[0], 2)\
            + math.pow(input_example[1] - self.center_circle[1], 2)\
                < math.pow(self.radius_circle,2):
            return 1
        else:
            return 0


    def generate_data(self):

        self.data = torch.FloatTensor(self.number_of_examples, self.number_of_features).uniform_(0,1)

        self.targets = torch.LongTensor(self.number_of_examples)
        index_targets = torch.arange(0, self.number_of_examples)

        self.targets = index_targets.apply_(lambda i: check_target(self.data[i]))

        return self.data, self.targets    
    
    def yield_data(self):
        
        if self.shuffle==True:
            
            shuffled_indexes = torch.randperm(self.number_of_examples)
            self.data_shuffled = self.data[shuffled_indexes]
            self.targets_shuffled = self.targets[shuffled_indexes]
            
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
                          self.targets.narrow(0, batch_start, self.batch_size)
                else:
                    yield self.data.narrow(0, batch_start, self.number_of_examples - batch_start),\
                          self.targets.narrow(0, batch_start, self.number_of_examples - batch_start)
        
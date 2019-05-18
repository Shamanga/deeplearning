import torch.nn as nn
import torch.nn.functional as F


class NN1(nn.Module):
    """
    Implementation of Neural Network:
    2 convulational and 2 linear layers, ReLU activation function applied
    
    Functions:
        - forward: feeding the specified network with the data x in the forward pass
        
    """
    def __init__(self):
        
        """
        Defining the functions to be used in building the neural nets' architecture
        """
        
        super(NN1, self).__init__()
        
        #Convolution 1
        self.conv1 = nn.Conv2d(2, 10, kernel_size=3)
        #Convolution 2
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        
        #Dropout for regularization
        self.conv2_drop = nn.Dropout2d()
        
        #Fully Connected 1
        self.fc1 = nn.Linear(20*2*2, 50)
        
        #Fully Connected 2
        self.fc2 = nn.Linear(50, 10)
        
        #Fully Connected 3
        self.fc3 = nn.Linear(10, 2)
        
    def forward(self, x):
        """
        The forward pass
        Args: 
            - x:   The input data
        Returns:
            - x: Tensor of size 2x1 showing the probabilities of the labels
        """
        # convolution 2, droupout, pooling, relu
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        
        # convolution 2, droupout, pooling, relu
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        # Resize
        x = x.view(-1, 20*2*2) 
        
        # Linear layer with activation
        x = F.relu(self.fc1(x))
        # Linear layer with activation
        x = F.relu(self.fc2(x))
        
        # Linear function
        x = self.fc3(x)
        
        # Rescale the values into [0,1]
        x = F.log_softmax(x)
       
        return x
    

  

    

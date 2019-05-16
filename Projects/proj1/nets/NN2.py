import torch.nn as nn
import torch.nn.functional as F
class NN2(nn.Module):
    """
    Implementation of Neural Network: with 2 subnetwork architecture to learn the additional information besides the targets
    2 convulational and 6 linear layers, ReLU activation function applied
    Functions:
        - forward: feeding the specified network with the data x in the forward pass
        
    """
    def __init__(self):
        """
        Defining the functions to be used in building the neural nets' architecture
        """
        super(NN2, self).__init__()
        #Convolution 1
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        #Convolution 2
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        
        #Dropout for regularization
        self.conv2_drop = nn.Dropout2d()
        
        #Fully Connected 1
        self.fc1 = nn.Linear(20*2*2, 50)
        
        #Fully Connected 2
        self.fc2 = nn.Linear(50, 10)
        
        #Fully Connected 3
        self.fc3 = nn.Linear(50, 30)
        
        #Fully Connected 4
        self.fc4 = nn.Linear(30, 20)
        
        #Fully Connected 5
        self.fc5 = nn.Linear(20, 10)
        
        #Fully Connected 6
        self.fc6 = nn.Linear(10, 2)

    def forward(self, data):
        """
        The forward pass iterates through the 2d data and for each applies the network to learn 
        both the digits' classes and targets.
        
        Args: 
            - data: The input data
        Returns:
            - digits: learned digits for the given data
            - targets: learned targets for the given data
        
        """
        digits = []
        rep_features = []
        #iterate through the input data( in our case we have 2 channel data)
        for i in range(2):
            x = data[:,i].view(data[:,0].shape[0],1,14,14)
            # convolution 1, pooling, relu
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            
            # convolution 2, droupout, pooling, relu
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            # Resize
            x = x.view(-1, 20*2*2)
            
            # store the representative features of each image before further processing 
            rep_features.append(self.fc1(x))
            
            # Linear function1, relu
            x = F.relu(self.fc1(x))
            
            # Dropout for regularization
            x = F.dropout(x, training=self.training)
            # Linear function 2
            x = self.fc2(x)
            
            # append the calculated digit for the input x 
            digits.append(F.log_softmax(x))
            
        # subtract the representative features of both images to get the difference   
        y = rep_features[0] - rep_features[1]
        # Linear function3, relu
        y = F.relu(self.fc3(y))
        # Linear function4, relu
        y = F.relu(self.fc4(y))
        # Linear function5, relu
        y = F.relu(self.fc5(y))
        # Linear function6
        y = self.fc6(y)
        # get the binary value as a target
        targets = F.log_softmax(y)
        return digits, targets
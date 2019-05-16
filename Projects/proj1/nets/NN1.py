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
        self.cnn1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        
        #Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2)
        
        #Dropout for regularization
        self.dropout = nn.Dropout(p=0.2)
        
        #Fully Connected 1
        self.fc1 = nn.Linear(16*14*14, 18)
        #Fully Connected 2
        self.fc2 = nn.Linear(18, 2)
        
    def forward(self, x):
        """
        The forward pass
        Args: 
            - x:   The input data
        Returns:
            - out: The output of the function
        
        """
        #Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        
        #Convolution 2
        
        out = F.relu(self.cnn2(out))
       
        #Resize
        out = out.view(out.size(0), -1)
        
        #Dropout
        out = self.dropout(out)
        
        #Fully connected 1
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = F.log_softmax(out)
        return out
    

  

    

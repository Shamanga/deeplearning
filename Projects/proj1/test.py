import torch
import torchvision
import warnings
import torch.nn as nn
from PIL import Image
from utils.utils import *
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.dataset import Dataset
from timeit import default_timer as timer
from nets.NN1 import NN1
from nets.NN2 import NN2
from nets.train_test import * 
from utils.MnistPairs import MnistPairs
%load_ext autoreload
%autoreload 2

#### Parameters ####
n_epochs = 25
iters = 10
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
#### Parameters ####

def main():
    """
    The function 
    - loads the train and test dataset according to given batchsizes 
    - trains and tests the first network
    - trains and tests the second network
    """
    #Loading dataset
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    train_dataset = MnistPairs('data/',train=True, transform=None)
    test_dataset = MnistPairs('data/',train=False, transform=None)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size_train, 
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size_test, 
                                              shuffle=False)
    
    print("\n Start of the Network 2 ")
    network2 = NN2()
    optimizer2 = optim.SGD(network2.parameters(), lr=learning_rate, momentum=momentum)
    
    print("The number of parameters of the network is ", count_parameters(network2))
    for i in range(iters):
        for epoch in range(1, n_epochs + 1):
            start = timer()
            train_losses_digits, train_losses_classes, train_acc_digits, train_acc_classes  = train2(epoch, network2,train_loader,optimizer2) 
            end = timer()
            print("Time needed to train ", end - start)
            test_loss_digits, test_loss_targets, acc_digits, acc_target = test2(network2, test_loader)
    
   
    print("Start of the Network 1 ")
    network1 = NN1()
    optimizer1 = optim.SGD(network1.parameters(), lr=learning_rate, momentum=momentum)
    
    print("The number of parameters of the network is ", count_parameters(network1))
    for i in range(iters):
        for epoch in range(1, n_epochs + 1):
            start = timer()
            train_loss, train_acc  = train1(epoch, network1, train_loader, optimizer1) 
            end = timer()
            print("Time needed to train ", end - start)
            test_loss, acc = test1(network1, test_loader)


if __name__ == '__main__':
    main()

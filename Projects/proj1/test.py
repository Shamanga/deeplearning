import torch
import torchvision
import warnings
import torch.nn as nn
from PIL import Image
import statistics as st
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
import matplotlib.pyplot as plt


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
    
    
    train_loss__ = []
    train_acc_digits__ = []
    train_acc_classes__ = []
    
    train_loss_ = []
    train_acc_digits_ = []
    train_acc_classes_= []

    test_loss__ = []
    test_acc_digits__ = []
    test_acc_classes__ = []
    avg_time2 = 0
    for i in range(iters):
        for epoch in range(1, n_epochs + 1):
            start = timer()
            train_loss, train_acc_digits, train_acc_classes  = train2(epoch, network2,train_loader,optimizer2) 
            end = timer()
            print("Time needed to train ", end - start)
            avg_time2 += end - start
            test_loss, test_acc_digits, test_acc_classes = test2(network2, test_loader)
            
            train_loss__ += train_loss
            train_acc_digits__ += train_acc_digits
            train_acc_classes__ += train_acc_classes
            
            train_loss_.append(mean(train_loss))
            train_acc_digits_.append(mean(train_acc_digits))
            train_acc_classes_.append(mean(train_acc_classes))
            
            test_loss__.append(test_loss)
            test_acc_digits__.append(test_acc_digits)
            test_acc_classes__.append(test_acc_classes)
            
    avg_time2 /=(iters * n_epochs)
    
    #plotting the results
    plt.figure()
    plt.plot(train_loss__, label = "train_loss__") 
    plt.ylabel("Negative log likelihood loss")
    plt.xlabel("Iterations")
    plt.title("Network 2 Classification Loss ")
    plt.show()
    
    plt.figure()
    plt.plot(train_acc_digits__, label = "train_acc_digits")
    plt.plot(train_acc_classes__, label = "train_acc_classes") 
    plt.legend()
    plt.ylabel("Accuracy")
    plt.xlabel("Iterations")
    plt.title("Network 2 Classification Accuracy ")
    plt.show()
    
    plt.figure()
    plt.plot(test_loss__, label = "test_loss") 
    plt.plot(train_loss_, label = "train_loss")
    plt.legend()
    plt.ylabel("Negative log likelihood loss")
    plt.xlabel("Number of seen testing data")
    plt.title("Network 2 Classification Loss ")
    plt.show()
    
    plt.figure()
    plt.plot(test_acc_digits__, label = "test_acc_digits")
    plt.plot(test_acc_classes__, label = "test_acc_classes")
    plt.plot(train_acc_digits_, label = "train_acc_digits")
    plt.plot(train_acc_classes_, label = "train_acc_classes")
    plt.legend()
    plt.ylabel("Accuracy")
    plt.xlabel("Number of seen data")
    plt.title("Network 2 Classification Accuracy")
    plt.show()
                      
    
    
    
   
    print("*************************** Start of the Network 1 ***************************")
    network1 = NN1()
    optimizer1 = optim.SGD(network1.parameters(), lr=learning_rate, momentum=momentum)
    
    train_loss__ = []
    train_acc__ = []
    
    train_loss_ = []
    train_acc_ = []

    test_loss__ = []
    test_acc__ = []
    avg_time1 = 0
    
    print("\n The number of parameters of the network is \n ", count_parameters(network1))
    for i in range(iters):
        t_loss = []
        t_acc = []
        test_loss_ = []
        test_acc_ = []
        for epoch in range(1, n_epochs + 1):
            start = timer()
            train_loss, train_acc  = train1(epoch, network1, train_loader, optimizer1) 
            end = timer()
            print("\n Time needed to train ", end - start)
            avg_time1 += end - start
            
            test_loss, test_acc = test1(network1, test_loader)
            
            train_loss__ += train_loss
            train_acc__ += train_acc_
            
            train_loss_.append(mean(train_loss))
            train_acc_.append(mean(train_acc))
            
            test_loss__.append(test_loss)
            test_acc__.append(test_acc)
            
    avg_time1 /= (iters * n_epochs)
    #plotting the results
    plt.figure()
    plt.plot(train_loss__, label = "train_loss__") 
    plt.ylabel("Cross Entropy loss")
    plt.legend()
    plt.xlabel("Iterations")
    plt.title("Network 1 Classification Loss ")
    plt.show()
    
    
    plt.figure()
    plt.plot(test_loss__, label = "test_loss") 
    plt.plot(train_loss_, label = "train_loss")
    plt.legend()
    plt.ylabel("Cross Entropy loss")
    plt.xlabel("Iterations")
    plt.title("Network 1 Classification Loss ")
    plt.show()
    
    plt.figure()
    plt.plot(test_acc_digits__, label = "test_acc_digits")
    plt.plot(test_acc_classes__, label = "test_acc_classes")
    plt.plot(train_acc_digits_, label = "train_acc_digits")
    plt.plot(train_acc_classes_, label = "train_acc_classes")
    plt.legend()
    plt.ylabel("Accuracy")
    plt.xlabel("Iterations")
    plt.title("Network 1 Classification Accuracy")
    plt.show()
    
    return avg_time2, train_acc_classes__, train_acc_digits__ , test_acc_classes__, test_acc_digits__, avg_time1, train_acc__, test_acc__
      
        
        
       
if __name__ == '__main__':
    avg_time2, train_acc_classes__, train_acc_digits__ , test_acc_classes__, test_acc_digits__, avg_time1, train_acc__, test_acc__ = main()
    print(" \n ************************ Statistics of Network 2 ************************ \n")
    print("\n Average time needed for training: ", avg_time2)
    print("\n Training: ","\n Average accuracy of labels: {}".format(st.mean(train_acc_classes__)),"\n Median accuracy of labels: {}".format(st.median(train_acc_classes__)),"\n Average accuracy on labels: {}".format(st.mean(train_acc_classes__)),"\n Maximum accuracy on labels: {}".format(max(train_acc_classes__)),"\n Minimum accuracy on labels: {}".format(min(train_acc_classes__)),"\n Spread of accuracy on labels: {}".format(st.stdev(train_acc_classes__)),"\n Average accuracy of digits: {}".format(st.mean(train_acc_digits__)),"\n Median accuracy of digits: {}".format(st.median(train_acc_digits__)),"\n Average accuracy on digits: {}".format(st.mean(train_acc_digits__)),"\n Maximum accuracy on digits: {}".format(max(train_acc_digits__)),"\n Minimum accuracy on digits: {}".format(min(train_acc_digits__)),"\n Spread of accuracy on digits: {}".format(st.stdev(train_acc_digits__)))
    print("\n Testing: ","\n Average accuracy of labels: {}".format(st.mean(test_acc_classes__)),"\n Median accuracy of labels: {}".format(st.median(test_acc_classes__)),"\n Average accuracy on labels: {}".format(st.mean(test_acc_classes__)),"\n Maximum accuracy on labels: {}".format(max(test_acc_classes__)),"\n Minimum accuracy on labels: {}".format(min(test_acc_classes__)),"\n Spread of accuracy on labels: {}".format(st.stdev(test_acc_classes__)),"\n Average accuracy of digits: {}".format(st.mean(test_acc_digits__)),"\n Median accuracy of digits: {}".format(st.median(test_acc_digits__)),"\n Average accuracy on digits: {}".format(st.mean(test_acc_digits__)),"\n Maximum accuracy on digits: {}".format(max(test_acc_digits__)),"\n Minimum accuracy on digits: {}".format(min(test_acc_digits__)),"\n Spread of accuracy on digits: {}".format(st.stdev(test_acc_digits__)))
    
    
    print("\n ************************ Statistics of Network 1 ************************ \n")
    print("\n Average time needed for training ", avg_time1)
    print("\n Training: ", "\n Average accuracy of labels: {}".format(st.mean(train_acc__)),"\n Median accuracy of labels: {}".format(st.median(train_acc__)),"\n Average accuracy on labels: {}".format(st.mean(train_acc__)),"\n Maximum accuracy on labels: {}".format(max(train_acc__)),"\n Minimum accuracy on labels: {}".format(min(train_acc__)),"\n Spread of accuracy on labels: {}".format(st.stdev(train_acc__)))
    print("\n Testing: ", "\n Average accuracy of labels: {}".format(st.mean(test_acc__)),"\n Median accuracy of labels: {}".format(st.median(test_acc__)), "\n Average accuracy on labels: {}".format(st.mean(test_acc__)),"\n Maximum accuracy on labels: {}".format(max(test_acc__)), "\n Minimum accuracy on labels: {}".format(min(test_acc__)), "\n Spread of accuracy on labels: {}".format(st.stdev(test_acc__)))
      
    
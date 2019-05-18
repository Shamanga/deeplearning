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
batch_size_train = 16
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(2019)
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
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size_test, 
                                              shuffle=True)
    # initialize lists for storing the performances
    train_loss_2 = [0] * n_epochs
    train_acc_digits_2 = [0]* n_epochs
    train_acc_classes_2= [0]* n_epochs
    
    test_loss_2 = [0] * n_epochs
    test_acc_digits_2 = [0] * n_epochs
    test_acc_classes_2 = [0] * n_epochs
    avg_time_2 = 0
    
    train_loss_1 = [0] * n_epochs
    train_acc_classes_1= [0]* n_epochs
    
    test_loss_1 = [0] * n_epochs
    test_acc_classes_1 = [0] * n_epochs
    avg_time_1 = 0
    # Test the architecture for 10 rounds to see get the whole picrture of performance
    
    for i in range(iters):
        
         # every time change the seed for random(consistent) initializations

        torch.manual_seed(2019+i)
        
        print("\n Start of the Network 2 ")
        # create the network
        network2 = NN2()
        # initialize the optimizer
        optimizer2 = optim.SGD(network2.parameters(), lr=learning_rate, momentum=momentum)
        # print the number of parameters
        print("The number of parameters of the network is ", count_parameters(network2))

        
        # initialize lists for storing the performance measures of each epoch
       
        train_loss__2 = []
        train_acc_digits__2 = []
        train_acc_classes__2 = []

        test_loss__2 = []
        test_acc_digits__2 = []
        test_acc_classes__2 = []
        # for getting the average time of the network 2 to train
        avg_time2 = 0


        
        for epoch in range(1, n_epochs + 1):
            # set timer
            start = timer()
            # train the network for train_loader data
            train_loss, train_acc_digits, train_acc_classes  = train2(epoch, network2,train_loader,optimizer2) 
            end = timer()
            print("Time needed to train ", end - start)
            # add up the time needed to train the net
            avg_time2 += end - start
            # get the test loss and accuracy
            test_loss, test_acc_digits, test_acc_classes = test2(network2, test_loader)            
            # get the train loss and accuracy (test function is used because train function loss results were different because of dropout)
            train_loss, train_acc_digits, train_acc_classes = test2(network2, train_loader)
            # store the obtained metrics
            train_loss__2.append(train_loss)
            train_acc_digits__2.append(train_acc_digits)
            train_acc_classes__2.append(train_acc_classes)

            test_loss__2.append(test_loss)
            test_acc_digits__2.append(test_acc_digits)
            test_acc_classes__2.append(test_acc_classes)
        # get the average time
        avg_time2 /= n_epochs
        avg_time_2 += avg_time2
        #######################plotting the losses and accuracies
        plt.figure()
        plt.plot(train_acc_digits__2, label = "train_acc_digits")
        plt.plot(train_acc_classes__2, label = "train_acc_classes") 
        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Iterations")
        plt.title("Network 2 Classification Accuracy ")
        plt.show()

        plt.figure()
        plt.plot(test_loss__2, label = "test_loss") 
        plt.plot(train_loss__2, label = "train_loss")
        plt.legend()
        plt.ylabel("Negative log likelihood loss")
        plt.xlabel("Number of seen testing data")
        plt.title("Network 2 Classification Loss ")
        plt.show()

        plt.figure()
        plt.plot(test_acc_digits__2, label = "test_acc_digits")
        plt.plot(train_acc_digits__2, label = "train_acc_digits")
        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Number of seen data")
        plt.title("Network 2 Classification Accuracy")
        plt.show()

        plt.figure()
        plt.plot(test_acc_classes__2, label = "test_acc_classes")
        plt.plot(train_acc_classes__2, label = "train_acc_classes")
        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Number of seen data")
        plt.title("Network 2 Classification Accuracy")
        plt.show()
         #######################
        
        # for each iteration add them up into the storage lists 
        
        train_loss_2 = [sum(x) for x in zip(train_loss_2, train_loss__2)]
        train_acc_digits_2 = [sum(x) for x in zip(train_acc_digits_2, train_acc_digits__2)]
        train_acc_classes_2= [sum(x) for x in zip(train_acc_classes_2, train_acc_classes__2)]
        
        test_loss_2 = [sum(x) for x in zip(test_loss_2, test_loss__2)]
        test_acc_digits_2 = [sum(x) for x in zip(test_acc_digits_2, test_acc_digits__2)]
        test_acc_classes_2= [sum(x) for x in zip(test_acc_classes_2, test_acc_classes__2)]
        


        print("*************************** Start of the Network 1 ***************************")
        # create the network
        network1 = NN1()
        # initialize the architecture
        optimizer1 = optim.SGD(network1.parameters(), lr=learning_rate, momentum=momentum)
        # initialize lists for storing the performance measures of each epoch
        
        train_loss__1 = []
        train_acc_classes__1 = []

        test_loss__1 = []
        test_acc_classes__1 = []
        avg_time1 = 0

        print("\n The number of parameters of the network is \n ", count_parameters(network1))
        #change the random seed into i
        torch.manual_seed(i)

        for epoch in range(1, n_epochs + 1):
            # set timer
            start = timer()
            # train the network for train_loader data
            train_loss, train_acc  = train1(epoch, network1, train_loader, optimizer1) 
            end = timer()
            print("\n Time needed to train ", end - start)
            # add up the time needed to train the net
            avg_time1 += end - start
             # get the loss and accuracy of test data
            test_loss, test_acc = test1(network1, test_loader)
            # get the loss and accuracy of train data
            train_loss, train_acc = test1(network1, train_loader)
            # store the obtained metrics
            test_loss__1.append(test_loss)
            test_acc_classes__1.append(test_acc)

            train_loss__1.append(train_loss)
            train_acc_classes__1.append(train_acc)
        # get the average time
        avg_time1 /= n_epochs
        avg_time_1 += avg_time1
        #######################plotting the losses and accuracies of after the epochs
        plt.figure()
        plt.plot(test_loss__1, label = "test_loss") 
        plt.plot(train_loss__1, label = "train_loss")
        plt.legend()
        plt.ylabel("Negative log likelihood loss")
        plt.xlabel("Number of seen testing data")
        plt.title("Network 1 Classification Loss ")
        plt.show()

        plt.figure()
        plt.plot(test_acc_classes__1, label = "test_acc_classes")
        plt.plot(train_acc_classes__1, label = "train_acc_classes")
        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Number of seen data")
        plt.title("Network 1 Classification Accuracy")
        plt.show()
        
        train_loss_1 = [sum(x) for x in zip(train_loss_1, train_loss__1)]
        train_acc_classes_1= [sum(x) for x in zip(train_acc_classes_1, train_acc_classes__1)]
        
        test_loss_1 = [sum(x) for x in zip(test_loss_1, test_loss__1)]
        test_acc_classes_1= [sum(x) for x in zip(test_acc_classes_1, test_acc_classes__1)]
    
    train_loss_2 = [ x / iters for x in train_loss_2]
    train_acc_digits_2 = [ x /iters for x in train_acc_digits_2]
    train_acc_classes_2 = [ x/iters for x in train_acc_classes_2]
    test_loss_2 = [ x/iters for x in test_loss_2]
    test_acc_digits_2 = [ x/iters for x in test_acc_digits_2]
    test_acc_classes_2 = [ x/iters for x in test_acc_classes_2]
    train_loss_1 = [ x/iters for x in train_loss_1]
    train_acc_classes_1 = [ x/iters for x in train_acc_classes_1]
    test_loss_1 = [ x/iters for x in test_loss_1]
    test_acc_classes_1 = [ x/iters for x in test_acc_classes_1]
    return avg_time_2/iters, train_acc_classes_2, train_acc_digits_2 , test_acc_classes_2, test_acc_digits_2, avg_time_1/iters, train_acc_classes_1, test_acc_classes_1
      
        
        
if __name__ == '__main__':
    avg_time2, train_acc_classes__, train_acc_digits__ , test_acc_classes__, test_acc_digits__, avg_time1, train_acc__, test_acc__ = main()
    print(" \n ************************ Statistics of Network 2 ************************ \n")
    print("\n Average time needed for training: ", avg_time2)
    print("\n Training: ","\n Average accuracy of labels: {}".format(mean(train_acc_classes__)),"\n Median accuracy of labels: {}".format(st.median(train_acc_classes__)),"\n Average accuracy on labels: {}".format(st.mean(train_acc_classes__)),"\n Maximum accuracy on labels: {}".format(max(train_acc_classes__)),"\n Minimum accuracy on labels: {}".format(min(train_acc_classes__)),"\n Spread of accuracy on labels: {}".format(st.stdev(train_acc_classes__)),"\n Average accuracy of digits: {}".format(st.mean(train_acc_digits__)),"\n Median accuracy of digits: {}".format(st.median(train_acc_digits__)),"\n Average accuracy on digits: {}".format(st.mean(train_acc_digits__)),"\n Maximum accuracy on digits: {}".format(max(train_acc_digits__)),"\n Minimum accuracy on digits: {}".format(min(train_acc_digits__)),"\n Spread of accuracy on digits: {}".format(st.stdev(train_acc_digits__)))
    print("\n Testing: ","\n Average accuracy of labels: {}".format(mean(test_acc_classes__)),"\n Median accuracy of labels: {}".format(st.median(test_acc_classes__)),"\n Average accuracy on labels: {}".format(st.mean(test_acc_classes__)),"\n Maximum accuracy on labels: {}".format(max(test_acc_classes__)),"\n Minimum accuracy on labels: {}".format(min(test_acc_classes__)),"\n Spread of accuracy on labels: {}".format(st.stdev(test_acc_classes__)),"\n Average accuracy of digits: {}".format(st.mean(test_acc_digits__)),"\n Median accuracy of digits: {}".format(st.median(test_acc_digits__)),"\n Average accuracy on digits: {}".format(st.mean(test_acc_digits__)),"\n Maximum accuracy on digits: {}".format(max(test_acc_digits__)),"\n Minimum accuracy on digits: {}".format(min(test_acc_digits__)),"\n Spread of accuracy on digits: {}".format(st.stdev(test_acc_digits__)))
    
    
    print("\n ************************ Statistics of Network 1 ************************ \n")
    print("\n Average time needed for training ", avg_time1)
    print("\n Training: ", "\n Average accuracy of labels: {}".format(st.mean(train_acc__)),"\n Median accuracy of labels: {}".format(st.median(train_acc__)),"\n Average accuracy on labels: {}".format(st.mean(train_acc__)),"\n Maximum accuracy on labels: {}".format(max(train_acc__)),"\n Minimum accuracy on labels: {}".format(min(train_acc__)),"\n Spread of accuracy on labels: {}".format(st.stdev(train_acc__)))
    print("\n Testing: ", "\n Average accuracy of labels: {}".format(st.mean(test_acc__)),"\n Median accuracy of labels: {}".format(st.median(test_acc__)), "\n Average accuracy on labels: {}".format(st.mean(test_acc__)),"\n Maximum accuracy on labels: {}".format(max(test_acc__)), "\n Minimum accuracy on labels: {}".format(min(test_acc__)), "\n Spread of accuracy on labels: {}".format(st.stdev(test_acc__)))
      
    
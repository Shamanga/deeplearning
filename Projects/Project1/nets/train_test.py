import torch
import torch
from utils.utils import *
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import warnings
from torch.utils.data.dataset import Dataset
import torch.optim as optim
from PIL import Image
from statistics import mean

def train1(epoch, network, train_loader, optimizer):
    """
    Train the defined network for the given data
    Args:
     - epoch: the number of times we pass through the dataset
     - network: defined architecture of layers and functions 
     - train_loader: loading the data with specified batch size
     
    Returns:
     - train_loss: the negative log likelihood loss of the target classification
     - train_acc: the accuracy of the classified targets 
     """
    # initialize the network for training
    network.train()
    
    # define lists to store Loss and accuracy after each batch data process
    train_loss = []
    train_acc = []
    
    # iterate trough the batched train data, train the model and optimize the loss
    for batch_idx, (data, target, classes) in enumerate(train_loader):
        
        #get the number of overall data
        total = target.size(0)
        
        #Clear the gradients
        optimizer.zero_grad()
        
        #Forward propagation 
        outputs = network(data)
        loss = F.nll_loss(outputs, target)
        
        #Backward propation
        loss.backward()
        
        #Updating gradients
        optimizer.step()
        
        #Obtaining predictions from max value
        _, predicted = torch.max(outputs.data, 1)
        
        #Calculate the number of correct answers
        correct = (predicted == target).sum().item()
        
        # store the loss and acc information for each batch
        train_loss.append(loss)
        train_acc.append(correct/total * 100)
        
        
        #Print loss and accuracy
        print('Epoch [{}/{}],Step [{}/{}],Loss: {:.4f} ,Acc_labels: {}/{} ({:.0f}%)'.format(epoch + 1, 25, batch_idx + 1, total, loss.item(), correct, total, correct/(total) * 100 ))
    return train_loss, train_acc

def test1(network, test_loader):
    """
    Test the defined network for the given data
    Args:
     
     - network:     trained network
     - test_loader: loading the test data with specified batch size
    
    Returns:
     - test_loss: the negative log likelihood loss of the target classification
     - acc: the accuracy of the classified targets 
    """
    # put the network into evaluation phase
    network.eval()
    
    # iterate trough the data, evaluate the network for the input, calculate the loss and store them
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target, classes) in enumerate(test_loader):
            
            total += classes[:,0].size(0)
            
            outputs = network(data)

            test_loss += F.nll_loss(outputs, target)

            #Obtaining predictions from max value
            _, predicted = torch.max(outputs.data, 1)
            #Calculate the number of correct answers
            correct += (predicted == target).sum().item()
    
    # obtain the accuracy by the number of correctly predicted labels      
    acc = 100. * correct /(total)
    
    # get the loss for each batch
    test_loss /= (batch_idx+1)
    
    print('\nTest set: Loss: {:.4f},Acc_labels: {}/{} {:.0f}%'.format(test_loss, correct, total, acc))
    return test_loss, acc
    
def train2(epoch, network,train_loader,optimizer):
    """
    Train the defined network for the given data
    Args:
     - epoch: the number of times we pass through the dataset
     - network: defined architecture of layers and functions 
     - train_loader: loading the data with specified batch size
     - optimizer: optimizer to minimize the loss 
     - alpha: weight between (0,1) specifying which loss(targets or digits) to consider more
     
    Returns:
     - train_losses_digits: the negative log likelihood loss of the digits classification
     - train_losses_classes: the negative log likelihood loss of the target classification
     - train_acc_digits: the accuracy of the classified digits 
     - train_acc_classes: the accuracy of the classified targets 
    """
    # initialize the network for training
    network.train()
    
    # define lists to store Loss and accuracy after each batch data process
    correct = 0
    train_loss = []
    train_acc_digits = []
    train_acc_targets = []

    for batch_idx, (data, target, classes) in enumerate(train_loader):
        #Total number of labels
        total = classes[:,0].size(0)
        
        #Clear the gradients
        optimizer.zero_grad()
        
        #Forward propagation 
        digits, labels = network(data)
        
        # negative log likelihood of image1 and image2 digit classification loss
        loss_digits = (F.nll_loss(digits[0], classes[:,0]) + F.nll_loss(digits[1], classes[:,1]))/2
        loss_target = F.nll_loss(labels, target)
        
        # get the final loss by convex combination of digits and targets losses
        loss = loss_digits + loss_target
        
        # Backward propogation
        loss.backward()
        #Updating the parameters
        optimizer.step()
        
        # Get the predicted digits
        pred_digits_1 = digits[0].data.max(1, keepdim=True)[1]
        pred_digits_2 = digits[1].data.max(1, keepdim=True)[1]

        # Get the count of correct predictions of digits
        correct_digits = (pred_digits_1.eq(classes[:,0].data.view_as(pred_digits_1)).sum() + pred_digits_2.eq(classes[:,1].data.view_as(pred_digits_2)).sum()).item()
        #Obtaining predictions from max value
        _, pred_label = torch.max(labels.data, 1)
        
        #Calculate the number of correct answers
        correct_target = (pred_label == target).sum().item()
        
        # store the loss and acc information for each batch
        train_loss.append(loss.item())
        train_acc_digits.append((correct_digits / (2*total)) * 100)
        train_acc_targets.append(correct_target / total * 100)
        
        #Print loss and accuracy
        print('Epoch [{}/{}],Step [{}/{}],Loss: {:.4f}, Acc_digits: {}/{} ({:.0f}%) ,Acc_labels: {}/{} ({:.0f}%)'.format(epoch + 1, 25, batch_idx + 1, total, loss.item(), correct_digits, 2*total, correct_digits/(2*total) * 100,correct_target, total, correct_target/total * 100 ))
    return train_loss, train_acc_digits, train_acc_targets  
    
    
def test2(network,test_loader):
    """
    Test the defined network for the given data
    Args:
     
     - network:     trained network
     - test_loader: loading the test data with specified batch size
     
     
    Returns:
     - test_loss_digits: the negative log likelihood loss of the digits classification
     - test_loss_classes: the negative log likelihood loss of the target classification
     - acc_digits: the accuracy of the classified digits 
     - acc_target: the accuracy of the classified targets 
    """
    # Put the network into Evaluation 
    
    network.eval()
    
    # iterate trough the data, evaluate the network for the input, calculate the loss and store them
    total = 0
    loss = 0
    correct_digits = 0
    correct_label = 0
    with torch.no_grad():
        for batch_idx, (data, target, classes) in enumerate(test_loader):
            digits, labels = network(data)
            
            #Total number of labels
            total += classes[:,0].size(0)
            # negative log likelihood of image1 and image2 digit classification loss
            test_loss_digits = (F.nll_loss(digits[0], classes[:,0])+ 
                                F.nll_loss(digits[1], classes[:,1]))/2
            # negative log likelihood of label classification
            test_loss_target = F.nll_loss(labels, target)
            
            # get the final loss by adding the loss of digits' and targets' 
            loss += test_loss_digits.item() + test_loss_target.item()

            pred_digits_1 = digits[0].data.max(1, keepdim=True)[1]
            pred_digits_2 = digits[1].data.max(1, keepdim=True)[1]

            correct_digits += (pred_digits_1.eq(classes[:,0].data.view_as(pred_digits_1)).sum() + pred_digits_2.eq(classes[:,1].data.view_as(pred_digits_2)).sum()).item()
            
            #Obtaining predictions from max value
            _, pred_label = torch.max(labels.data, 1)
            #Calculate the number of correct answers
            correct_label += (pred_label == target).sum().item()
    # take the mean of the accuracies     
    acc_target = 100. * correct_label /(total)
    acc_digits = 100. * correct_digits /( 2* total)
    loss /= (batch_idx+1)
    
    print('\nTest set: Loss: {:.4f},Acc_digits: {}/{} ({:.0f}%),Acc_labels: {}/{} {:.0f}%'.format(loss, correct_digits, 2*total, acc_digits, correct_label, total, acc_target))
    return loss, acc_digits, acc_target
    
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
    network.train()
    train_loss = []
    train_acc = []
    for batch_idx, (data, target, classes) in enumerate(train_loader):
        #Clear the gradients
        optimizer.zero_grad()
        
        #Forward propagation 
        outputs = network(data)
        
        criterion = nn.CrossEntropyLoss()
        
        loss = criterion(outputs, target)
        #Create instance of optimizer (Adam)
        #optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
        
        #Backward propation
        loss.backward()
        
        #Updating gradients
        optimizer.step()
        
        #Total number of labels
        total = target.size(0)
        
        #Obtaining predictions from max value
        _, predicted = torch.max(outputs.data, 1)
        
        #Calculate the number of correct answers
        correct = (predicted == target).sum().item()
        
        # store the loss and acc information for each batch
        train_loss.append(loss.item())
        train_acc.append(correct/total * 100)
        
        
        #Print loss and accuracy
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
              .format(epoch + 1, 25, batch_idx + 1, len(train_loader),
                         loss.item(), correct/total * 100))
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
    network.eval()
    
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target, classes in test_loader:
            outputs = network(data)

            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(outputs, target)

            #Obtaining predictions from max value
            _, predicted = torch.max(outputs.data, 1)
            #Calculate the number of correct answers
            correct += (predicted == target).sum().item()
            
    acc = 100. * correct /(len(test_loader.dataset))
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy_labels: {:.0f}% '.format(
    test_loss, acc))
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
    network.train()
    
    correct = 0
    train_losses_digits = []
    train_losses_targets = []
    train_acc_digits = []
    train_acc_targets = []

    for batch_idx, (data, target, classes) in enumerate(train_loader):
        #Clear the gradients
        optimizer.zero_grad()
        #Forward propagation 
        digits, labels = network(data)
        
        # negative log likelihood of image1 and image2 digit classification loss
        loss_digit_1 = F.nll_loss(digits[0], classes[:,0])
        loss_digit_2 = F.nll_loss(digits[1], classes[:,1])

        # get the mean of the loss
        loss_digits = (loss_digit_1 + loss_digit_2)/2
        # get the loss of the target classification
        loss_target = F.nll_loss(labels, target)
        
        # get the final loss by convex combination of digits and targets losses
        loss = loss_digits + loss_target
        
        # Backward propogation
        loss.backward()
        #Updating the parameters
        optimizer.step()
        
        #Total number of labels
        total = classes[:,0].size(0)

        #Obtaining predictions from max value
        _, pred_digits_1 = torch.max(digits[0].data, 1)
        _, pred_digits_2 = torch.max(digits[1].data, 1)
        
        #Calculate the number of correct answers
        correct_digits = (pred_digits_1 == classes[:,0]).sum().item() + (pred_digits_2 == classes[:,1]).sum().item()
        
        #Obtaining predictions from max value
        _, pred_label = torch.max(labels.data, 1)

        #Calculate the number of correct answers
        correct_target = (pred_label == target).sum().item()
        
        # store the loss and acc information for each batch
        train_losses_digits.append(loss_digits.item())
        train_losses_targets.append(loss_target.item())
        train_acc_digits.append((correct_digits / (2*total)) * 100)
        train_acc_targets.append(correct_target / total * 100)
        
        #Print loss and accuracy
        print('Epoch [{}/{}],Step [{}/{}],Loss_digits: {:.4f},Loss_targets: {:.4f},Accuracy_digits: {:.2f}%,Accuracy_labels: {:.2f}%'
                 .format(epoch + 1, 25, batch_idx + 1, len(train_loader),
                         loss_digits.item(),loss_target.item(),(correct_digits / (2*total)) * 100, correct_target/total * 100))
    return train_losses_digits, train_losses_targets, train_acc_digits, train_acc_targets  
    
    
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
    network.eval()
    test_loss_digits = 0
    test_loss_target = 0
    correct_digits = 0
    correct_label = 0
    
    with torch.no_grad():
        for data, target, classes in test_loader:
            digits, labels = network(data)

            test_loss_digits += 0.5*(F.nll_loss(digits[0], classes[:,0], size_average=False).item()+ F.nll_loss(digits[1], classes[:,1], size_average=False).item())
            test_loss_target += F.nll_loss(labels, target, size_average=False).item()

            pred_digits_1 = digits[0].data.max(1, keepdim=True)[1]
            pred_digits_2 = digits[1].data.max(1, keepdim=True)[1]

            correct_digits += pred_digits_1.eq(classes[:,0].data.view_as(pred_digits_1)).sum() + pred_digits_2.eq(classes[:,1].data.view_as(pred_digits_2)).sum()
            #Obtaining predictions from max value
            _, pred_label = torch.max(labels.data, 1)
    
            #Calculate the number of correct answers
            correct_label += (pred_label == target).sum().item()
    acc_target = 100. * correct_label /(len(test_loader.dataset))
    acc_digits = 100. * correct_digits /( 2* len(test_loader.dataset))
    test_loss_digits /= len(test_loader.dataset)
    test_loss_target /= len(test_loader.dataset)
    print('\nTest set:Avg. loss_digits: {:.4f},Avg. loss_targets: {:.4f},Accuracy_digits: {}/{} ({:.0f}%)\n,Accuracy_labels: {:.0f}%'.format(
    test_loss_digits, test_loss_target, correct_digits, 2*len(test_loader.dataset),
    acc_digits, acc_target))
    return test_loss_digits, test_loss_target, acc_digits, acc_target
    
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


class MnistPairs(Dataset):
    
    training_file = 'training.pt'
    test_file = 'test.pt'

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data
    
    @property
    def train_classes(self):
        warnings.warn("train_classes has been renamed classes")
        return self.classes
    
    @property
    def test_classes(self):
        warnings.warn("test_classes has been renamed classes")
        return self.classes
    
    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')
    
    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.processed_folder, self.test_file))
    
    def mnist_to_pairs(self, nb, input, target):
        input = torch.functional.F.avg_pool2d(input, kernel_size = 2)
        a = torch.randperm(input.size(0))
        a = a[:2 * nb].view(nb, 2)
        input = torch.cat((input[a[:, 0]], input[a[:, 1]]), 1)
        classes = target[a]
        target = (classes[:, 0] <= classes[:, 1]).long()
        return input, target, classes

    def generate_pair_sets(self, nb):
        if self.train:
            dataset = datasets.MNIST(self.root + '/mnist/', train = True, download = True)
            dataset_input = dataset.train_data.view(-1, 1, 28, 28).float()
            dataset_target = dataset.train_labels
        
        else:
            dataset = datasets.MNIST(self.root + '/mnist/', train = False, download = True)
            dataset_input = dataset.test_data.view(-1, 1, 28, 28).float()
            dataset_target = dataset.test_labels

        return self.mnist_to_pairs(nb, dataset_input, dataset_target)
    
    def __init__(self, root, nb = 1000, train=True, transform=None, target_transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        
        if not self._check_exists():
            raise RuntimeError('Dataset not found.')
            
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        
        self.data ,self.targets, self.classes = self.generate_pair_sets(nb)
        
    
    
    def __getitem__(self, index):
        
        img, target, classes = self.data[index], int(self.targets[index]), self.classes[index]
        
        img_1 = Image.fromarray(img[0].numpy(), mode='L')
        img_2 = Image.fromarray(img[1].numpy(), mode='L')
        
        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
            
            img = torch.stack((img_1.reshape(img_1.shape[1],img_1.shape[2]),img_2.reshape(img_2.shape[1],img_2.shape[2])),dim=0)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, classes

    def __len__(self):
        return len(self.data)
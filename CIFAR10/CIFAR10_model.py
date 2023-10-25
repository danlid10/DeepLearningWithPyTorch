import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
import os

""" The CIFAR-10 dataset is a collection of 60,000 32x32 color images grouped into 10 classes,
with 50,000-image training set and a 10,000-image test set """

# Parameters setup
train_log_path = "CIFAR10_train_log.txt"
model_path = "CIFAR10_model.pth"
num_classes = 10
learning_rate = 0.001
num_epochs = 50
batch_size = 4

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defining the model
class ConvNeuralNet(nn.Module):
    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        """ 
        Size after a convolutional layer can be expressed as follows:
        [(W − K + 2P)/S] + 1 
        where:  W - the input size, K - the Kernel size, P - padding, S - stride
        """  

    def forward(self, x):
        # init shape: n, 3, 32, 32
        x = self.pool(self.relu(self.conv1(x)))     # conv1: n, 6, 28, 28  -> pooling: n, 6, 14, 14
        x = self.pool(self.relu(self.conv2(x)))     # conv1: n, 16, 10, 10 -> poolong: n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)                  # flatten: n, 16*5*5 = 400
        x = self.relu(self.fc1(x))                  # fc1: n, 120
        x = self.relu(self.fc2(x))                  # fc2: n, 84
        x = self.fc3(x)                             # fc1: n, 10 
        return x
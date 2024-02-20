import torch
import torch.nn as nn

""" The CIFAR-10 dataset is a collection of 60,000 32x32 color images grouped into 10 classes,
with 50,000-image training set and a 10,000-image test set """

# Parameters setup
LEARNING_RATE = 0.001
BATCH_SIZE = 4
NUM_EPOCHS = 50
TRAIN_LOG_PATH = "CIFAR10_train_log.txt"
MODEL_PATH = "CIFAR10_model.pth"
USE_TENSORBOARD = False

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defining the model
class ConvNeuralNet(nn.Module):
    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        self.num_classes = 10
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

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
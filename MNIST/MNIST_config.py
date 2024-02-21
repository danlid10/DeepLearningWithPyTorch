import torch
import torch.nn as nn

""" The MNIST dataset consists of 60,000 training images and 10,000 testing images,
with each image being a grayscale 28x28 pixel representation of a handwritten digit (0 through 9). """

# Parameters setup
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 3
TRAIN_LOG_PATH = "MNIST_train_log.txt"
TEST_LOG_PATH = "MNIST_test_log.txt"
MODEL_PATH = "MNIST_model.pth"
USE_TENSORBOARD = False

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defining the model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(28 * 28, 112)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(112, 10)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x
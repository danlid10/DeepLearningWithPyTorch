import torch
import torch.nn as nn
import json

""" The MNIST dataset consists of 60,000 training images and 10,000 testing images,
with each image being a grayscale 28x28 pixel representation of a handwritten digit (0 through 9). """

with open("config.json", "r") as jsonfile:
    config = json.load(jsonfile)

# Defining the model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.input_size = 28 * 28
        self.num_classes = 10
        self.hidden_size = config["hidden_size"]
        self.l1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x
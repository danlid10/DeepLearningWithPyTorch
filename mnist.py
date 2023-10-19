import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

""" The MNIST dataset consists of 60,000 training images and 10,000 testing images,
with each image being a grayscale 28x28 pixel representation of a handwritten digit (0 through 9). """

# Load MNIST dataset
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

# Parameters setup
input_size = train_data.data.size(1) * train_data.data.size(2)
hidden_size = 112
num_classes = 10
learning_rate = 0.001
num_epochs = 5
batch_size = 64

# Data loaders setup
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

# Defining the model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_Szie = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, input):
        output = self.l1(input)
        output = self.relu(output)
        output = self.l2(output)
        return output

model = NeuralNet(input_size, hidden_size, num_classes)

# Defining loss and optimiser
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
total_steps = len(train_loader)
with open("training_log.txt", 'w') as fp:
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(train_loader):

            features = features.view(features.size(0), -1)
            output = model(features)
            loss = criterion(output, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            fp.write(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}\n')

# Testing the model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for features, labels in test_loader:

        features = features.view(features.size(0), -1)
        outputs = model(features)
        predicted = torch.argmax(outputs.data, dim=1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the {n_samples} test images: {acc} %')


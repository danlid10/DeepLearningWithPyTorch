import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from datetime import datetime

""" The MNIST dataset consists of 60,000 training images and 10,000 testing images,
with each image being a grayscale 28x28 pixel representation of a handwritten digit (0 through 9). """

# Parameters setup
train_log_path = "MNIST_train_log.txt"
model_path = "MNIST_model.pth"
force_train = True
input_size = 28 * 28
hidden_size = 112
num_classes = 10
learning_rate = 0.001
num_epochs = 3
batch_size = 64

# Load MNIST dataset
train_data = datasets.MNIST(root="data", train=True,
                            download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root="data", train=False,
                           download=True, transform=transforms.ToTensor())

# Data loaders setup
train_loader = DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

# Defining the model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_Szie = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x


model = NeuralNet(input_size, hidden_size, num_classes)

if os.path.exists(model_path) and not force_train:
    # Loading the model
    model.load_state_dict(torch.load(model_path)) 
    model.eval()
    print("Model loaded")
else: 
    # Defining loss and optimiser
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    # Training the model
    start_time = datetime.now()
    total_steps = len(train_loader)
    with open(train_log_path, 'w') as f:
        f.write(f"Training log from {start_time}\n")
        for epoch in range(num_epochs):
            for i, (features, labels) in enumerate(train_loader):

                features = features.view(features.size(0), -1)
                output = model(features)
                loss = criterion(output, labels)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                f.write(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}\n')

    end_time = datetime.now()
    torch.save(model.state_dict(), model_path)
    print(f"Training completed in {end_time - start_time}")

# Testing the model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_samples = [0] * num_classes
    n_class_correct = [0] * num_classes

    for features, labels in test_loader:
        features = features.view(features.size(0), -1)
        outputs = model(features)
        predicted = torch.argmax(outputs.data, dim=1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for pred, label in zip(predicted, labels):
            n_class_samples[label] += 1
            if pred == label:
                n_class_correct[label] += 1

    for i in range(num_classes):
        class_acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of class {i}: {class_acc:.3f} %')

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc:.3f} %')

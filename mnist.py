import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

""" The MNIST dataset consists of 60,000 training images and 10,000 testing images,
with each image being a grayscale 28x28 pixel representation of a handwritten digit (0 through 9). """

input_size = 28*28
hidden_size = 100
num_classes = 10
learning_rate = 0.001
num_epochs = 3
batch_size = 80


train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())

test_dataset = datasets.MNIST(root="data", train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)


model = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, num_classes))
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):

        output = model(features)
        loss = criterion(output, labels)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if (i + 1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')


# Testing the model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for features, labels in test_loader:

        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the {n_samples} test images: {acc} %')


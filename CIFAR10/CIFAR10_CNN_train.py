import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime

""" The CIFAR-10 dataset is a collection of 60,000 32x32 color images grouped into 10 classes,
with 50,000-image training set and a 10,000-image test set """

# Parameters setup
train_log_path = "CIFAR10_train_log.txt"
model_path = "CIFAR10_model.pth"
num_classes = 10
learning_rate = 0.001
num_epochs = 20
batch_size = 8

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

def main():
    # Load MNIST dataset
    train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms.ToTensor())
  
    # Data loaders setup
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    model = ConvNeuralNet()

    # Defining loss and optimiser
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(model.parameters(), lr=learning_rate)

    # Training the model
    start_time = datetime.now()
    total_steps = len(train_loader)
    with open(train_log_path, 'w') as f:
        f.write(f"Training log from {start_time}\n")
        for epoch in range(num_epochs):
            for i, (features, labels) in enumerate(train_loader):

                output = model(features)
                loss = criterion(output, labels)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                f.write(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}\n')

    end_time = datetime.now()
    torch.save(model.state_dict(), model_path)
    print(f"Training completed in {end_time - start_time}")

if __name__ == "__main__":
    main()
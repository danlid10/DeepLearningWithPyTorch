import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import json
from MNIST_model import NeuralNet

with open("config.json", "r") as jsonfile:
    config = json.load(jsonfile)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset
train_transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)) 
                                ])
train_data = datasets.MNIST(root="data", train=True, download=True, transform=train_transforms)

# Data loaders setup
train_loader = DataLoader(dataset=train_data, batch_size=config["batch_size"], shuffle=True)

model = NeuralNet().to(device)

writer = SummaryWriter()
# Loading example data and model to TensorBoard
examples = iter(train_loader)
features, labels = next(examples)
img_grid = torchvision.utils.make_grid(features)
writer.add_image('MNIST images', img_grid)
writer.add_graph(model, features)

# Defining loss and optimiser
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=config["learning_rate"])

# Training the model
start_time = datetime.now()
total_steps = len(train_loader)
running_loss = 0.0
with open(config["train_log_path"], 'w') as f:
    f.write(f"Training log from {start_time}\n")
    print("Training started")
    for epoch in tqdm(range(config["num_epochs"]), desc="Epoch"):
        for i, data in enumerate(tqdm(train_loader, desc="Step", leave=False)):
            
            features, labels = data
            features = features.view(features.size(0), -1).to(device)
            labels = labels.to(device)
            
            output = model(features)
            loss = criterion(output, labels)
            running_loss += loss.item()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if (i + 1) % 100 == 0:
                writer.add_scalar("Training loss", running_loss / 100, epoch * total_steps + i)
                running_loss = 0.0

            f.write(f'Epoch [{epoch+1}/{config["num_epochs"]}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}\n')

    writer.close()
    end_time = datetime.now()
    training_time = end_time - start_time
    f.write(f"Training completed in {training_time}")

torch.save(model.state_dict(), config["model_path"])
print(f"Training completed in {training_time}, model saved as '{config['model_path']}'")


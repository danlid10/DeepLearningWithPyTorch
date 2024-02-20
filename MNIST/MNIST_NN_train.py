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
import os
import MNIST_config

# Load MNIST dataset
train_transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)) 
                                ])
train_data = datasets.MNIST(root="data", train=True, download=True, transform=train_transforms)

# Data loader setup
train_loader = DataLoader(dataset=train_data, batch_size=MNIST_config.BATCH_SIZE, shuffle=True)

model = MNIST_config.NeuralNet().to(MNIST_config.DEVICE)

if MNIST_config.USE_TENSORBOARD:
    writer = SummaryWriter()
    # Loading example data and model to TensorBoard
    examples = iter(train_loader)
    features, labels = next(examples)
    img_grid = torchvision.utils.make_grid(features)
    writer.add_image('MNIST images', img_grid)
    writer.add_graph(model, features.view(features.size(0), -1))

# Defining loss and optimiser
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=MNIST_config.LEARNING_RATE)

# Training the model
start_time = datetime.now()
total_steps = len(train_loader)
running_loss = 0.0
os.makedirs('logs', exist_ok=True)
log_path = os.path.join('logs', f'{start_time.strftime("%Y%m%d-%H%M%S")}_{MNIST_config.TRAIN_LOG_PATH}')

with open(log_path, 'w') as f:
    f.write(f"Training log from {start_time}, Device: {MNIST_config.DEVICE}\n")
    print("Training started")
    for epoch in tqdm(range(MNIST_config.NUM_EPOCHS), desc="Epoch"):
        for i, data in enumerate(tqdm(train_loader, desc="Step", leave=False)):
            
            features, labels = data
            features = features.view(features.size(0), -1).to(MNIST_config.DEVICE)
            labels = labels.to(MNIST_config.DEVICE)
            
            output = model(features)
            loss = criterion(output, labels)
            running_loss += loss.item()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if MNIST_config.USE_TENSORBOARD and (i + 1) % 100 == 0:
                writer.add_scalar("Training loss", running_loss / 100, epoch * total_steps + i)
                running_loss = 0.0

            f.write(f'Epoch [{epoch+1}/{MNIST_config.NUM_EPOCHS}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}\n')

    if MNIST_config.USE_TENSORBOARD:
        writer.close()

    end_time = datetime.now()
    training_time = end_time - start_time
    f.write(f"Training completed in {training_time}")

torch.save(model.state_dict(), MNIST_config.MODEL_PATH)
print(f"Training completed in {training_time}, model saved as '{MNIST_config.MODEL_PATH}'")


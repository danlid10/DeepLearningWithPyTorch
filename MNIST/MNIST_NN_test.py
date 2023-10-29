import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import json
from MNIST_model import NeuralNet

with open("config.json", "r") as jsonfile:
    config = json.load(jsonfile)

if not os.path.exists(config["model_path"]):
    print("[ERROR] Model not found, exiting...")
    exit()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset
test_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                            ])
test_data = datasets.MNIST(root="data", train=False, download=True, transform=test_transforms)

# Data loaders setup
test_loader = DataLoader(dataset=test_data, batch_size=config["batch_size"])

model = NeuralNet()

# Loading the model
model.load_state_dict(torch.load(config["model_path"], map_location=device)) 
model.eval()
print("Model loaded")

# Testing the model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_samples = [0] * model.num_classes
    n_class_correct = [0] * model.num_classes

    for features, labels in test_loader:

        features = features.view(features.size(0), -1).to(device)
        labels = labels.to(device)
        
        outputs = model(features)
        predicted = torch.argmax(outputs.data, dim=1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for pred, label in zip(predicted, labels):
            n_class_samples[label] += 1
            if pred == label:
                n_class_correct[label] += 1

    for i in range(model.num_classes):
        class_acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of class {i}: {class_acc:.3f} %')

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc:.3f} %')



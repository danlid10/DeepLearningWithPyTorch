import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import CIFAR10_config
  
if not os.path.exists(CIFAR10_config.MODEL_PATH):
    print("[ERROR] Model not found, exiting...")
    exit()

# Load MNIST test dataset
test_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=test_transforms)

# Data loader setup
test_loader = DataLoader(dataset=test_data, batch_size=CIFAR10_config.BATCH_SIZE)

# TensorBoard writer
writer = SummaryWriter()

# Loading the model
model = CIFAR10_config.ConvNeuralNet()
model.load_state_dict(torch.load(CIFAR10_config.MODEL_PATH, map_location=device)) 
model.eval()
print(f"Model loaded to {CIFAR10_config.DEVICE}")

# Testing the model
classes = test_data.classes
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_samples = [0] * model.num_classes
    n_class_correct = [0] * model.num_classes
    class_probs = []
    class_label = []

    for features, labels in test_loader:

        features = features.to(CIFAR10_config.DEVICE)
        labels = labels.to(CIFAR10_config.DEVICE)

        outputs = model(features)

        class_probs_batch = [F.softmax(output, dim=0) for output in outputs]
        class_probs.append(class_probs_batch)
        class_label.append(labels)

        predicted = torch.argmax(outputs.data, dim=1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for pred, label in zip(predicted, labels):
            n_class_samples[label] += 1
            if pred == label:
                n_class_correct[label] += 1
                
    if CIFAR10_config.USE_TENSORBOARD:
        test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
        test_label = torch.cat(class_label)

    for i in range(model.num_classes):
        class_acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of class {classes[i]}: {class_acc:.3f} %')
        if CIFAR10_config.USE_TENSORBOARD:
            # Tensorboard PR curve
            tensorboard_truth = test_label == i
            tensorboard_probs = test_probs[:, i]
            writer.add_pr_curve(classes[i], tensorboard_truth, tensorboard_probs, global_step=0)
   
    if CIFAR10_config.USE_TENSORBOARD:
        writer.close()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc:.3f} %')


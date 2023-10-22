import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from CIFAR10_CNN_train import *

def main():

    if not os.path.exists(model_path):
        print("[ERROR] Model not found, exiting...")
        exit()

    # Load MNIST test dataset
    test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transforms.ToTensor())

    # Data loader setup
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

    model = ConvNeuralNet()

    # Loading the model
    model.load_state_dict(torch.load(model_path)) 
    model.eval()
    print("Model loaded")

    # Testing the model
    classes = test_data.classes
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_samples = [0] * num_classes
        n_class_correct = [0] * num_classes

        for features, labels in test_loader:
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
            print(f'Accuracy of class {classes[i]}: {class_acc:.3f} %')

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc:.3f} %')

if __name__ == "__main__":
    main()
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

training_transforms = transforms.Compose([
        transforms.RandomResizedCrop(28),
        transforms.RandomRotation(45),
        #transforms.ToTensor()
])

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=training_transforms
)

feature, label = training_data[4]
print(label)
plt.imshow(feature)
plt.show()
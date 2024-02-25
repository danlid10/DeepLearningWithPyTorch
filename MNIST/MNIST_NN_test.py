import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import MNIST_config

# Load MNIST dataset
test_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                            ])
test_data = datasets.MNIST(root="data", train=False, download=True, transform=test_transforms)

# Data loader setup
test_loader = DataLoader(dataset=test_data, batch_size=MNIST_config.BATCH_SIZE)

writer = SummaryWriter()

# Loading the model
model = MNIST_config.NeuralNet()
model.load_state_dict(torch.load(MNIST_config.MODEL_PATH, map_location=MNIST_config.DEVICE)) 
model.eval()
print(f"Model loaded to {MNIST_config.DEVICE}")

start_time = datetime.now()
os.makedirs('logs', exist_ok=True)
log_path = os.path.join('logs', f'{start_time.strftime("%Y%m%d-%H%M%S")}_{MNIST_config.TEST_LOG_PATH}')
print("Model summary:")
modelsum = summary(model, (1, 28 * 28))

# Testing the model
print("Testing started.")
classes = test_data.classes
num_classes = len(classes)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_samples = [0] * num_classes
    n_class_correct = [0] * num_classes
    class_probs = []
    class_label = []

    for features, labels in test_loader:

        features = features.view(features.size(0), -1).to(MNIST_config.DEVICE)
        labels = labels.to(MNIST_config.DEVICE)
        
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

        test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
        test_label = torch.cat(class_label)
    
    
    with open(log_path, 'w', encoding='utf-8') as f:

        f.write(f"Testing log from {start_time}, Device: {MNIST_config.DEVICE}\n")
        f.write(f"Model summary:\n {str(modelsum)}\n")

        for i in range(num_classes):

            class_acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of class "{classes[i]}": {class_acc:.3f} %')
            f.write(f'Accuracy of class "{classes[i]}": {class_acc:.3f} %\n')

            # TensorBoard PR curve
            tensorboard_truth = test_label == i
            tensorboard_probs = test_probs[:, i]
            writer.add_pr_curve(classes[i], tensorboard_truth, tensorboard_probs, global_step=0)

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc:.3f} %')
        f.write(f'Accuracy of the network: {acc:.3f} %\n')

writer.close()



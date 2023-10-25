from MNIST_model import *

def main():
    
    if not os.path.exists(model_path):
        print("[ERROR] Model not found, exiting...")
        exit()

    # Load MNIST dataset
    test_transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                                ])
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=test_transforms)

    # Data loaders setup
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

    model = NeuralNet(input_size, hidden_size, num_classes)

    # Loading the model
    model.load_state_dict(torch.load(model_path, map_location=device)) 
    model.eval()
    print("Model loaded")

    # Testing the model
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_samples = [0] * num_classes
        n_class_correct = [0] * num_classes

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

        for i in range(num_classes):
            class_acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of class {i}: {class_acc:.3f} %')

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc:.3f} %')


if __name__ == "__main__":
    main()
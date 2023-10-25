from MNIST_model import *
 
def main():

    # Load MNIST dataset
    train_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)) 
                                    ])
    train_data = datasets.MNIST(root="data", train=True, download=True, transform=train_transforms)

    # Data loaders setup
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    # Defining loss and optimiser
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    # Training the model
    start_time = datetime.now()
    total_steps = len(train_loader)
    with open(train_log_path, 'w') as f:
        f.write(f"Training log from {start_time}\n")
        print("Training started")
        for epoch in tqdm(range(num_epochs), desc="Epoch"):
            for i, (features, labels) in enumerate(train_loader):

                features = features.view(features.size(0), -1).to(device)
                labels = labels.to(device)
                
                output = model(features)
                loss = criterion(output, labels)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                f.write(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}\n')

        end_time = datetime.now()
        training_time = end_time - start_time
        f.write(f"Training completed in {training_time}")

    torch.save(model.state_dict(), model_path)
    print(f"Training completed in {training_time}, model saved as '{model_path}'")

if __name__ == "__main__":
    main()
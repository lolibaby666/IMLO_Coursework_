import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vgg16_model import VGG16  # Make sure this is your VGG16 class definition

# Step 1: Set up the necessary configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 2: Define transformations for test data (same as the one used during training)
transform = transforms.Compose([
    transforms.ToTensor()
])

# Step 3: Load the CIFAR-10 test dataset
data_root = './data'  # Location of the CIFAR-10 dataset
testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

# Step 4: Create DataLoader for test dataset
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

# Step 5: Load the trained model
model = VGG16(weight_decay=0.0001)  # Ensure this matches the model architecture during training
model.to(device)

# Load the trained model weights (assuming the model was saved as 'vgg16_cifar10.pth')
model_path = './vgg16_cifar10.pth'
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

# Step 6: Define the loss function (same as used during training)
criterion = nn.CrossEntropyLoss()

# Step 7: Evaluate the model on the test set
def evaluate(model, testloader, criterion, device):
    test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():  # No need to compute gradients during testing
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    # Calculate the average loss and accuracy
    test_loss /= len(testloader)
    test_accuracy = 100 * correct_test / total_test

    return test_loss, test_accuracy

# Step 8: Run the evaluation
test_loss, test_accuracy = evaluate(model, testloader, criterion, device)

# Step 9: Print the test results
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Step 10: Visualization of Correct and Incorrect Predictions
def visualize_predictions(model, testloader, num_samples=10):
    model.eval()  # Set model to evaluation mode
    data_iter = iter(testloader)
    
    # Get a batch of test data
    inputs, labels = next(data_iter)
    inputs, labels = inputs.to(device), labels.to(device)

    # Predict the outputs
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)

    # Create lists for correct and incorrect predictions
    correct_samples = []
    incorrect_samples = []

    for i in range(len(labels)):
        true_label = testset.classes[labels[i]]
        pred_label = testset.classes[predicted[i]]
        if true_label == pred_label:
            correct_samples.append((inputs[i], true_label, pred_label))
        else:
            incorrect_samples.append((inputs[i], true_label, pred_label))
        
        # Stop early if we have enough correct/incorrect samples
        if len(correct_samples) >= num_samples and len(incorrect_samples) >= num_samples:
            break

    # Visualizing Correct Predictions
    fig = plt.figure(figsize=(12, 6))
    for i in range(min(num_samples, len(correct_samples))):
        ax = fig.add_subplot(2, 5, i+1)
        img = correct_samples[i][0].cpu().numpy().transpose(1, 2, 0)
        ax.imshow(np.clip(img, 0, 1))
        
        true_label = correct_samples[i][1]
        pred_label = correct_samples[i][2]
        ax.set_title(f"True: {true_label}\nPred: {pred_label}")
        ax.axis('off')

    # Save the figure with correct predictions
    plt.tight_layout()
    plt.savefig('correct_predictions.png', dpi=300)

    # Visualizing Incorrect Predictions
    fig = plt.figure(figsize=(12, 6))
    for i in range(min(num_samples, len(incorrect_samples))):
        ax = fig.add_subplot(2, 5, i+1)
        img = incorrect_samples[i][0].cpu().numpy().transpose(1, 2, 0)
        ax.imshow(np.clip(img, 0, 1))
        
        true_label = incorrect_samples[i][1]
        pred_label = incorrect_samples[i][2]
        ax.set_title(f"True: {true_label}\nPred: {pred_label}")
        ax.axis('off')

    # Save the figure with incorrect predictions
    plt.tight_layout()
    plt.savefig('incorrect_predictions.png', dpi=300)
    plt.show()

# Step 11: Run the visualization function
visualize_predictions(model, testloader, num_samples=10)

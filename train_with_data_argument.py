import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vgg16_model import VGG16
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numpy as np

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Step 1: Data Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomRotation(15)      # Random rotation up to 15 degrees
])

# Load CIFAR-10 training and validation datasets
data_root = './data'
trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)

# Split the training set into training and validation sets (90% for training, 10% for validation)
trainset, validset = torch.utils.data.random_split(trainset, [45000, 5000])

# Create DataLoader for training and validation datasets
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
validloader = DataLoader(validset, batch_size=128, shuffle=False, num_workers=4)

# Step 2: Model Initialization (Using VGG16)
model = VGG16(weight_decay=0.0001)

# Step 3: Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()  # Cross-Entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate of 0.001

# Step 4: Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Step 5: Training Loop
device = torch.device("cpu")
model.to(device)

def train(model, trainloader, validloader, criterion, optimizer, scheduler, epochs=10):
    model.train()  # Set the model to training mode
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Wrap the trainloader with tqdm for the progress bar
        for i, (inputs, labels) in enumerate(tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}', ncols=100, leave=False)):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero gradients before the backward pass
            outputs = model(inputs)  # Forward pass
            
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update model weights
            
            running_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)
        
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()  # Set the model to evaluation mode during validation
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():  # No need to compute gradients during validation
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate validation accuracy
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss /= len(validloader)
        val_accuracy = 100 * correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Print Epoch Results
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        # Step the scheduler at the end of each epoch to reduce learning rate
        scheduler.step()
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Step 6: Train the Model
train_losses, val_losses, train_accuracies, val_accuracies = train(model, trainloader, validloader, criterion, optimizer, scheduler, epochs=100)

# Step 7: Save the model after training
model_save_path = './vgg16_cifar10_data_argument.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Step 8: Plot Training and Validation Loss Curves
plt.figure(figsize=(12, 6))

# Training and Validation Loss curve in one graph
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Training and Validation Accuracy curve in one graph
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()

# Save the figure to a file
plt.tight_layout()

# Save the entire figure
plt.savefig('training_validation_plots.png', dpi=300) 
plt.show()




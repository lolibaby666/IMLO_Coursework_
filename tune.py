import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from vgg16_model import VGG16
from tqdm import tqdm
import random
import numpy as np

# ----------------------- #
#       UTILITIES         #
# ----------------------- #

def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    """Force CPU usage only."""
    return torch.device("cpu")

def get_dataloaders(batch_size=128):
    """Prepares and returns training and validation dataloaders with data augmentation."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
    ])
    data_root = './data'
    dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    trainset, valset = random_split(dataset, [45000, 5000])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader, valloader

# ----------------------- #
#   MODEL + OPT HELPERS   #
# ----------------------- #

def get_model(name, activation_fn, weight_decay=0.0001):
    """
    Returns a model instance by name.
    Supports 'vgg16' and 'resnet18'.
    """
    if name == "vgg16":
        return VGG16(activation_fn=activation_fn, weight_decay=weight_decay)
    elif name == "resnet18":
        model = models.resnet18(pretrained=False, num_classes=10)
        return model
    else:
        raise ValueError(f"Unknown model: {name}")

def get_optimizer(name, parameters, lr):
    """
    Returns an optimizer instance given name and parameters.
    """
    if name == "adam":
        return optim.Adam(parameters, lr=lr)
    elif name == "sgd":
        return optim.SGD(parameters, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def get_loss(name):
    """
    Returns the loss function by name.
    """
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")

# ----------------------- #
#     TRAINING LOOP       #
# ----------------------- #

def train_and_evaluate(model, trainloader, valloader, criterion, optimizer, device, epochs=5):
    """
    Trains the model and evaluates on the validation set.
    Returns final validation accuracy.
    """
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

    val_accuracy = 100 * correct / total
    return val_accuracy

# ----------------------- #
#     MAIN TUNING LOOP    #
# ----------------------- #

def main():
    set_seed()
    device = get_device()
    print(f"Using device: {device}")
    trainloader, valloader = get_dataloaders()

    # Define hyperparameter space
    model_names = ["vgg16", "resnet18"]
    optimizers = ["adam", "sgd"]
    losses = ["cross_entropy"]
    lrs = [0.001, 0.0005]
    activations = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU()
    }

    # Start grid search
    results = []
    for model_name in model_names:
        for act_name, act_fn in activations.items():
            for loss_name in losses:
                for opt_name in optimizers:
                    for lr in lrs:
                        print(f"\nTesting: Model={model_name}, Act={act_name}, Loss={loss_name}, Optim={opt_name}, LR={lr}")
                        model = get_model(model_name, act_fn)
                        model.to(device)
                        criterion = get_loss(loss_name)
                        optimizer = get_optimizer(opt_name, model.parameters(), lr)
                        acc = train_and_evaluate(model, trainloader, valloader, criterion, optimizer, device)
                        print(f"Val Accuracy: {acc:.2f}%")
                        results.append({
                            "model": model_name,
                            "activation": act_name,
                            "loss": loss_name,
                            "optimizer": opt_name,
                            "lr": lr,
                            "val_accuracy": acc
                        })

    # Print top results
    print("\nTop Configurations:")
    results.sort(key=lambda x: x["val_accuracy"], reverse=True)
    for r in results[:5]:
        print(r)

if __name__ == "__main__":
    main()

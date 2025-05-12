import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from vgg16_model import VGG16
from resnet_model import ResNet
import random
import numpy as np
from losses import FocalLoss

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    """Force CPU-only usage."""
    return torch.device("cpu")

def get_train_val_loader(batch_size=128):
    """
    Loads CIFAR-10 once, splits it into distinct train and validation sets,
    applies augmentation to training only, and returns two DataLoaders.
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load full dataset with initial transform
    data_root = './data'
    full_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)

    # Perform a single consistent split
    train_size = 45000
    val_size = 5000
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Override transform for validation data
    val_subset.dataset.transform = val_transform

    # Create data loaders
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, valloader

def get_model(name, activation_fn=nn.ReLU(), weight_decay=0.0001):
    """
    Create and return a model instance.

    Args:
        name (str): Model name ('vgg16' or 'resnet').
        activation_fn (type): Activation function class, e.g., nn.ReLU.
        weight_decay (float): Weight decay for regularization (used in VGG16).

    Returns:
        nn.Module: Instantiated model.
    """
    if name == "vgg16":
        return VGG16(activation_fn=activation_fn, weight_decay=weight_decay)
    elif name == "resnet":
        return ResNet(activation_fn=activation_fn)
    else:
        raise ValueError(f"Unknown model: {name}")

def get_optimizer(optimizer_name, parameters, lr):
    """
    Return optimizer by name.
    """
    if optimizer_name == "adam":
        return optim.Adam(parameters, lr=lr)
    elif optimizer_name == "sgd":
        return optim.SGD(parameters, lr=lr, momentum=0.9)
    elif optimizer_name == "adamw":
        return optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_loss(loss_name):
    """
    Return loss function by name.
    """
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "focal":
        return FocalLoss(gamma=2.0)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

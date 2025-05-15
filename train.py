import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from utils import set_seed, get_device, get_train_val_loader, get_model, get_optimizer, get_loss


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Run one full training epoch.
    Args:
        model: The PyTorch model to train.
        dataloader: Dataloader for the training set.
        criterion: Loss function.
        optimizer: Optimizer to update weights.
        device: CPU or GPU device.

    Returns:
        Average training loss for the epoch.
    """
    model.train() # Set model to training mode (enables dropout, batchnorm updates)
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # --- Accumulate total loss and accuracy stats ---
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100 * correct / total


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on validation data.

    Args:
        model: The trained model.
        dataloader: Dataloader for validation.
        criterion: Loss function.
        device: CPU.

    Returns:
        Tuple of average validation loss and accuracy.
    """
    model.eval() # Set model to evaluation mode (disables dropout/batchnorm update)
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Turn off gradient computation for evaluation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # --- Accumulate total loss and accuracy stats ---
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, 100 * correct / total


def train(model, trainloader, valloader, criterion, optimizer, device, config):
    """
    Full training loop with early stopping and metric tracking.

    Args:
        model: Model to train.
        trainloader: Training DataLoader.
        valloader: Validation DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: CPU/GPU.
        config: Dict containing 'epochs', 'patience'.

    Returns:
        Tuple of (train_losses, val_losses, train_accs, val_accs)
    """
    best_acc = 0.0
    no_improve = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        start = time.time()

        # --- Train and Validate ---
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, valloader, criterion, device)

        # --- Logging ---
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # --- Early stopping check ---
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Time for epoch: {round(time.time() - start, 2)}s")

    return train_losses, val_losses, train_accuracies, val_accuracies


def plot_metrics(train_losses, val_losses, train_accs, val_accs, save_path="training_plot.png"):
    """
    Plot and save training/validation loss and accuracy curves.

    Args:
        train_losses: List of training losses.
        val_losses: List of validation losses.
        train_accs: List of training accuracies.
        val_accs: List of validation accuracies.
        save_path: Path to save the plot image.
    """
    plt.figure(figsize=(12, 5))

    # --- Loss Curve ---
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # --- Accuracy Curve ---
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Training plots saved to {save_path}")

def main():
    """
    Main training script for CIFAR-10 using VGG16 or ResNet.
    """
    set_seed()
    device = get_device()

    # --- Configurations ---
    config = {
        "model": "vgg16",             
        "activation": nn.ReLU(),       
        "loss": "cross_entropy",     
        "optimizer": "adam",
        "lr": 0.001,
        "batch_size": 128,
        "epochs": 100,
        "patience": 30,
        "save_path": "model.pth"
    }

    # --- Data ---
    trainloader, valloader = get_train_val_loader(config["batch_size"])

    # --- Model ---
    model = get_model(config["model"], activation_fn=config["activation"])

    # --- Loss and Optimizer ---
    criterion = get_loss(config["loss"])
    optimizer = get_optimizer(config["optimizer"], model.parameters(), config["lr"])

    # --- Training --- 
    total_start = time.time()
    train_losses, val_losses, train_accs, val_accs = train(
        model, trainloader, valloader, criterion, optimizer, device, config
    )
    total_end = time.time()
    print(f"\n Total training time: {round(total_end - total_start, 2)} seconds")

    # --- Save model ---
    torch.save(model.state_dict(), config["save_path"])
    print(f"Model saved to {config['save_path']}")

    # --- Plot metrics ---
    plot_metrics(train_losses, val_losses, train_accs, val_accs, save_path="training_metrics.png")


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from utils import get_device, get_model, get_loss, set_seed


def load_test_data(batch_size):
    """
    Load and return the CIFAR-10 test DataLoader and class names.

    Args:
        batch_size (int): Batch size for test DataLoader.

    Returns:
        testloader (DataLoader): DataLoader for the CIFAR-10 test set.
        class_names (list): List of class names in CIFAR-10.
    """
    transform = transforms.ToTensor()
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return testloader, testset.classes


def evaluate_and_collect(model, testloader, criterion, device):
    """
    Evaluate the model on the test set and collect predictions and corresponding images.

    Args:
        model (nn.Module): Trained model.
        testloader (DataLoader): Test dataset loader.
        criterion (Loss): Loss function used during evaluation.
        device (torch.device): Target device (CPU).

    Returns:
        avg_loss (float): Average loss on test set.
        accuracy (float): Overall test accuracy (%).
        all_preds (np.array): Predicted labels.
        all_labels (np.array): True labels.
        all_images (list): Corresponding image tensors.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels, all_images = [], [], []

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Store predictions and inputs for visualization
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_images.extend(inputs.cpu())  # Keep image tensors for visualization

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), all_images


def save_classification_report(y_true, y_pred, class_names, filename="classification_report.csv"):
    """
    Save classification metrics (precision, recall, F1-score) to a CSV.

    Args:
        y_true (array): Ground-truth labels.
        y_pred (array): Predicted labels.
        class_names (list): List of class names.
        filename (str): Output CSV filename.
    """
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    df.to_csv(filename, float_format="%.4f")
    print(f"Classification report saved to {filename}")


def plot_prediction_samples(images, labels, preds, class_names, title, filename, num_samples=10):
    """
    Plot and save a grid of correct or incorrect predictions.

    Args:
        images (list): List of image tensors.
        labels (list): True label indices.
        preds (list): Predicted label indices.
        class_names (list): List of class names.
        title (str): Title for the plot.
        filename (str): File path to save the image.
        num_samples (int): Number of images to visualize.
    """
    fig = plt.figure(figsize=(12, 5))
    shown = 0
    for i in range(len(images)):
        if shown >= num_samples:
            break
        if class_names[labels[i]] == class_names[preds[i]] and "Correct" in title:
            pass
        elif class_names[labels[i]] != class_names[preds[i]] and "Incorrect" in title:
            pass
        else:
            continue

        ax = fig.add_subplot(2, 5, shown + 1)
        img = images[i].numpy().transpose(1, 2, 0)
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(f"T: {class_names[labels[i]]}\nP: {class_names[preds[i]]}")
        ax.axis('off')
        shown += 1

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"{title} saved to {filename}")

def plot_confusion_matrix(y_true, y_pred, class_names, filename="confusion_matrix.png"):
    """
    Generate and save a confusion matrix plot.

    Args:
        y_true (array): Ground-truth labels.
        y_pred (array): Predicted labels.
        class_names (list): Class names to label axes.
        filename (str): Output image filename.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(xticks_rotation=90, cmap='Blues', ax=ax)  # <-- colorbar=True by default
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {filename}")



# ----------------------------
# Main Testing Logic
# ----------------------------

def main():
    """
    Main testing routine:
    - Loads test data and model
    - Evaluates test set
    - Saves metrics (loss, accuracy, classification report, confusion matrix)
    - Saves example visualizations of correct/incorrect predictions
    """
    set_seed()
    device = get_device()

    # --- Configuration dictionary ---
    config = {
        "model": "vgg16",
        "activation": nn.ReLU(),
        "loss": "cross_entropy",
        "batch_size": 128,
        "model_path": "model.pth",
        "num_visualize": 10,
        "prefix": "test"
    }

    # --- Load Data ---
    testloader, class_names = load_test_data(config["batch_size"])

    # --- Load Model ---
    model = get_model(config["model"], activation_fn=config["activation"])
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    model.to(device)
    criterion = get_loss(config["loss"])
    print(f"Loaded model from {config['model_path']}")

    # --- Evaluate ---
    test_loss, test_acc, y_pred, y_true, images = evaluate_and_collect(model, testloader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

    # --- Detailed Classification Report ---
    print("\nDetailed Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    save_classification_report(y_true, y_pred, class_names, filename=f"{config['prefix']}_report.csv")

    # --- Confusion Matrix ---
    plot_confusion_matrix(y_true, y_pred, class_names, filename=f"{config['prefix']}_confusion_matrix.png")

    # --- Prediction Visualization ---
    plot_prediction_samples(images, y_true, y_pred, class_names,
                            "Correct Predictions", f"{config['prefix']}_correct.png", num_samples=config["num_visualize"])
    plot_prediction_samples(images, y_true, y_pred, class_names,
                            "Incorrect Predictions", f"{config['prefix']}_incorrect.png", num_samples=config["num_visualize"])


if __name__ == "__main__":
    main()

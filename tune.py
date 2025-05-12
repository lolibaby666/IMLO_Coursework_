import torch
import torch.nn as nn
import time
import pandas as pd
from tqdm import tqdm
from utils import (
    set_seed, get_device, get_train_val_loader,
    get_model, get_optimizer, get_loss
)

def train_and_evaluate(model, trainloader, valloader, criterion, optimizer, device, epochs=1, patience=20):
    """
    Trains the model with early stopping and returns:
    - Best validation accuracy
    - Time taken
    - Epoch at which training stopped
    """
    model.to(device)
    best_acc = 0.0
    best_epoch = 0
    no_improve_count = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
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

        val_acc = 100 * correct / total

        # Early stopping logic
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    duration = time.time() - start_time
    return best_acc, duration, best_epoch


def main():
    set_seed()
    device = get_device()

    # Define hyperparameter search space
    model_names = ["vgg16", "resnet"]
    activations = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU()
    }
    optimizers = ["adam", "sgd", "adamw"]
    losses = ["cross_entropy","focal"]
    lrs = [0.001, 0.0005]
    batch_sizes = [64, 128]

    results = []

    # Loop through all combinations
    for batch_size in batch_sizes:
        trainloader, valloader = get_train_val_loader(batch_size=batch_size)
        for model_name in model_names:
            for act_name, act_fn in activations.items():
                for optimizer_name in optimizers:
                    for loss_name in losses:
                        for lr in lrs:
                            print(f"\n▶ Testing: Model={model_name}, Act={act_name}, Optim={optimizer_name}, "
                                  f"Loss={loss_name}, LR={lr}, Batch={batch_size}")
                            model = get_model(model_name, activation_fn=act_fn)
                            criterion = get_loss(loss_name)
                            optimizer = get_optimizer(optimizer_name, model.parameters(), lr)

                            # Train and evaluate
                            best_acc, duration, stop_epoch = train_and_evaluate(
                                model, trainloader, valloader, criterion, optimizer, device,
                                epochs=30, patience=5
                            )

                            print(f"Done: Val Acc = {best_acc:.2f}%, Stopped at epoch {stop_epoch}, Time = {round(duration, 2)}s")

                            # Record results
                            results.append({
                                "model": model_name,
                                "activation": act_name,
                                "optimizer": optimizer_name,
                                "loss": loss_name,
                                "lr": lr,
                                "batch_size": batch_size,
                                "val_accuracy": best_acc,
                                "stop_epoch": stop_epoch,
                                "duration_sec": round(duration, 2)
                            })

    # Sort and show top results
    results.sort(key=lambda x: x["val_accuracy"], reverse=True)
    print("\n Top Configurations:")
    for r in results[:5]:
        print(r)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("tuning_results.csv", index=False)
    print("\n Saved all results to tuning_results.csv")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_one_epoch(model, dataloader, criterion, optimizer, device="cpu"):
    # Standard training loop - returns (train_loss, train_acc)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    train_acc = 100.0 * correct / total
    return avg_loss, train_acc


def evaluate(model, dataloader, criterion, device="cpu"):
    # Standard evaluation loop - returns (val_loss, val_acc)
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    val_acc = 100.0 * correct / total
    return avg_loss, val_acc


def train_and_save_best(
    model, train_loader, valid_loader, criterion, optimizer, num_epochs=5, device="cpu"
):
    """
    Trains the model for the specified number of epochs,
    evaluating after each epoch. Saves the model if the
    validation accuracy improves.
    """

    best_val_acc = 0.0  # Track the highest validation accuracy
    best_model_path = "best_model.pth"

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        # 1) Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # 2) Validate
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%")

        # 3) Check if this is the best validation accuracy so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 4) Save the best model
            torch.save(model.state_dict(), best_model_path)
            print(
                f"New best model saved at epoch {epoch+1} "
                f"with Val Acc: {best_val_acc:.2f}%"
            )

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model is stored at: {best_model_path}")

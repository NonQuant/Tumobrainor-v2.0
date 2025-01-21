import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TumobrainorDataset
from model import AttentionResNet50


def train_one_epoch(model, dataloader, criterion, optimizer, device="cpu"):
    # Standard training loop - returns (train_loss, train_acc)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images.view(-1, 3, 224, 224))
        loss = criterion(outputs, torch.argmax(labels.view(32, 4), dim=1).long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # _, predicted = outputs.max(1)
        # correct += predicted.eq(labels).sum().item()
        # total += labels.size(0)
        predicted = torch.argmax(outputs, dim=1).data

        # calculate the amount of correct predictions in batch
        correct = (predicted == torch.argmax(labels.view(32, 4), dim=1)).sum()
        # add batch corrects to the total amount in training epoch

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
            outputs = model(images.view(-1, 3, 224, 224))
            loss = criterion(outputs, torch.argmax(labels.view(32, 4), dim=1).long())

            running_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1).data
            correct = (predicted == torch.argmax(labels.view(32, 4), dim=1)).sum()
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pretend you have numpy arrays (X_train, y_train, ...) ready
    # X_* shape: (N, 3, H, W), y_* shape: (N,) with values {1..4}
    # For demonstration, here we'll just create dummy data:
    # (Replace these with your real dataset arrays)
    N_train, N_valid, N_test = 32, 8, 8
    X_train = torch.randn(N_train, 3, 224, 224)
    y_train = torch.randint(1, 5, (N_train,))  # 4-class labels in {1..4}

    X_valid = torch.randn(N_valid, 3, 224, 224)
    y_valid = torch.randint(1, 5, (N_valid,))

    X_test = torch.randn(N_test, 3, 224, 224)
    y_test = torch.randint(1, 5, (N_test,))

    # Create datasets
    train_set = TumobrainorDataset(X_train, y_train)
    valid_set = TumobrainorDataset(X_valid, y_valid)
    test_set = TumobrainorDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(
        train_set, batch_size=4, shuffle=True, pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        valid_set, batch_size=4, shuffle=False, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=4, shuffle=False, pin_memory=True, drop_last=True
    )

    # Build the attention-based ResNet model
    model = AttentionResNet50(num_classes=4, freeze_backbone=True).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    # Train for a few epochs (example: 2 epochs)
    # for epoch in range(2):
    #     print(f"\nEpoch [{epoch+1}/2]")
    #     train_one_epoch(model, train_loader, criterion, optimizer, device)
    #     evaluate(model, valid_loader, criterion, device)

    train_and_save_best(model, train_loader, valid_loader, criterion, optimizer)

    print("\nFinal test evaluation:")
    evaluate(model, test_loader, criterion, device)


if __name__ == "__main__":
    main()

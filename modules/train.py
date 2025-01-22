import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import AttentionResNet50


# Print iterations progress
def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def train_one_epoch(model, dataloader, criterion, optimizer, epoch, device="cpu"):
    # Standard training loop - returns (train_loss, train_acc)
    model.train()
    running_loss = 0.0
    correct = 0
    total_batches = len(dataloader)
    print(len(dataloader))

    epoch_correct = 0

    epoch_start = time.time()

    for batch_num, (images, labels) in enumerate(dataloader):
        batch_start = time.time()

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
        epoch_correct += correct

        batch_end = time.time() - batch_start
        estimated_epoch_end = batch_end * (total_batches - batch_num) / 60  # in minutes

        printProgressBar(
            batch_num + 1,
            len(dataloader),
            prefix=f"Epoch {epoch} | Batch {(batch_num + 1)}/{total_batches}",
            suffix=f"{estimated_epoch_end:.2f} minutes remaining",
            length=50,
        )

    epoch_end = time.time() - epoch_start
    avg_loss = running_loss / total_batches
    train_acc = epoch_correct.item() * 100 / (4 * 8 * batch_num)
    print(
        f"Epoch {epoch} | Batch {(batch_num + 1) * 4}\nAccuracy: {train_acc:2.2f} | Loss: {avg_loss:2.4f} | Duration: {epoch_end / 60:.2f} minutes"
    )
    return avg_loss, train_acc


def evaluate(model, dataloader, criterion, epoch, device="cpu"):
    # Standard evaluation loop - returns (val_loss, val_acc)
    model.eval()
    running_loss = 0.0
    correct = 0
    epoch_correct = 0
    total = 0

    with torch.no_grad():
        total_batches = len(dataloader)
        for batch_num, (images, labels) in enumerate(dataloader):
            batch_start = time.time()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.view(-1, 3, 224, 224))
            loss = criterion(outputs, torch.argmax(labels.view(32, 4), dim=1).long())

            running_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1).data
            correct = (predicted == torch.argmax(labels.view(32, 4), dim=1)).sum()
            epoch_correct += correct
            total += labels.size(0)

            batch_end = time.time() - batch_start
            estimated_epoch_end = (
                batch_end * (total_batches - batch_num) / 60
            )  # in minutes

            printProgressBar(
                batch_num + 1,
                len(dataloader),
                prefix=f"Validation Epoch {epoch} | Batch {(batch_num + 1)}",
                suffix=f"{estimated_epoch_end:.2f} minutes remaining",
                length=50,
            )

    avg_loss = running_loss / len(dataloader)
    val_acc = epoch_correct.item() * 100 / (4 * 8 * batch_num)
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
        print("-" * 50)
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        # 1) Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, device
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # 2) Validate
        val_loss, val_acc = evaluate(model, valid_loader, criterion, epoch, device)
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

    # loading all dataloaders
    with open("./dataset/train_loader.pickle", "rb") as f:
        train_loader = pickle.load(f)
    with open("./dataset/valid_loader.pickle", "rb") as f:
        valid_loader = pickle.load(f)
    with open("./dataset/test_loader.pickle", "rb") as f:
        test_loader = pickle.load(f)

    # Build the attention-based ResNet model
    model = AttentionResNet50(num_classes=4, freeze_backbone=True).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    num_epochs = 1

    # Train for a few epochs (example: 2 epochs)
    # for epoch in range(num_epochs):
    #     print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    #     train_one_epoch(model, train_loader, criterion, optimizer, epoch + 1, device)
    #     evaluate(model, valid_loader, criterion, epoch + 1, device)

    train_and_save_best(
        model, train_loader, valid_loader, criterion, optimizer, num_epochs, device
    )

    print("\nFinal test evaluation:")
    evaluate(model, test_loader, criterion, "Final", device)


if __name__ == "__main__":
    main()

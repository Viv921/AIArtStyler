# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import config
import model
import dataset


def train_step(model, loader, optimizer, loss_fn, device):
    """
    Performs one training epoch.
    """
    model.train()
    total_loss = 0.0
    correct_preds = 0
    total_samples = 0

    loop = tqdm(loader, desc="Training", leave=True)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_preds += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        loop.set_postfix(
            loss=loss.item(),
            acc=(correct_preds / total_samples) * 100
        )

    epoch_loss = total_loss / total_samples
    epoch_acc = (correct_preds / total_samples) * 100
    return epoch_loss, epoch_acc


def validate_step(model, loader, loss_fn, device):
    """
    Performs one validation epoch.
    """
    model.eval()
    total_loss = 0.0
    correct_preds = 0
    total_samples = 0

    loop = tqdm(loader, desc="Validating", leave=True)
    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            loop.set_postfix(
                loss=loss.item(),
                acc=(correct_preds / total_samples) * 100
            )

    epoch_loss = total_loss / total_samples
    epoch_acc = (correct_preds / total_samples) * 100
    return epoch_loss, epoch_acc


def main():
    """
    Main training script.
    """
    print("Starting training...")
    print(f"Device: {config.DEVICE}")

    train_loader, val_loader, test_loader, class_names = dataset.get_dataloaders()

    if train_loader is None:
        return

    print(f"Data loaded. Training on {len(train_loader.dataset)} images.")

    clf_model = model.create_model()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf_model.parameters(), lr=config.LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.NUM_EPOCHS} ---")

        train_loss, train_acc = train_step(
            clf_model, train_loader, optimizer, loss_fn, config.DEVICE
        )
        print(f"Epoch {epoch + 1} Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        val_loss, val_acc = validate_step(
            clf_model, val_loader, loss_fn, config.DEVICE
        )
        print(f"Epoch {epoch + 1} Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(clf_model.state_dict(), config.MODEL_PATH)
            print(f"New best model saved! Val Acc: {val_acc:.2f}%")

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    print("Loading best model and evaluating on test set...")
    clf_model.load_state_dict(torch.load(config.MODEL_PATH))

    test_loss, test_acc = validate_step(
        clf_model, test_loader, loss_fn, config.DEVICE
    )
    print(f"Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    print(f"Model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
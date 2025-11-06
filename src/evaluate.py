# evaluate.py

import torch
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import config
import model
import dataset


def evaluate_model():
    """
    Loads the trained model and evaluates it on the test set.
    Generates a classification report and a confusion matrix.
    """

    print("Loading test data...")
    # Get the test_loader and class_names
    # We don't need train_loader or val_loader here
    _, _, test_loader, class_names = dataset.get_dataloaders()

    if test_loader is None:
        print("Failed to load data. Exiting.")
        return

    if len(class_names) != config.NUM_CLASSES:
        print(f"Warning: config.NUM_CLASSES is {config.NUM_CLASSES} but dataset has {len(class_names)} classes.")
        # Update config.NUM_CLASSES to match dataset
        config.NUM_CLASSES = len(class_names)

    print("Loading saved model...")
    # Initialize the model architecture
    clf_model = model.create_model(num_classes=config.NUM_CLASSES)

    # Load the saved weights
    try:
        clf_model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at {config.MODEL_PATH}")
        print("Please run train.py first to train and save a model.")
        return
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        print(
            "This might be due to a mismatch between the saved model and the current model.py (e.g., ResNet vs. EfficientNet).")
        return

    clf_model.to(config.DEVICE)
    clf_model.eval()

    all_preds = []
    all_labels = []

    print("Running evaluation on test set...")
    loop = tqdm(test_loader, desc="Evaluating", leave=True)
    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            # Forward pass
            outputs = clf_model(images)

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)

            # Store predictions and labels to CPU
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n--- Evaluation Results ---")

    # 1. Classification Report
    print("Classification Report:")
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=3
    )
    print(report)

    # 2. Confusion Matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)

    # Plot using Seaborn
    plt.figure(figsize=(15, 12))
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    sn.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    # Save the figure
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path)
    print(f"Confusion matrix saved to {plot_path}")


if __name__ == "__main__":
    evaluate_model()
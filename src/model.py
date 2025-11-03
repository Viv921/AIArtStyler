# model.py

import torch
import torch.nn as nn
from torchvision import models
import config


def create_model(num_classes=config.NUM_CLASSES):
    """
    Loads a pre-trained ResNet50 model and adapts it for fine-tuning.
    """

    # 1. Load pre-trained ResNet50
    # We use the modern weights argument for clarity
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # 2. Freeze parameters for fine-tuning
    for param in model.parameters():
        param.requires_grad = False

    # 3. Replace the final layer
    # Get the number of input features to the original FC layer
    num_ftrs = model.fc.in_features

    # Create a new FC layer with the correct number of output classes
    model.fc = nn.Linear(num_ftrs, num_classes)

    # 4. Move the model to the configured device
    model = model.to(config.DEVICE)

    print(f"Model: ResNet50 (fine-tuning all layers)")
    print(f"Output classes: {num_classes}")
    print(f"Device: {config.DEVICE}")

    return model
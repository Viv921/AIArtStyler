# model.py

import torch
import torch.nn as nn
from torchvision import models
import config


def create_model(num_classes=config.NUM_CLASSES):
    """
    Loads a pre-trained EfficientNet_B3 model and adapts it for fine-tuning.
    """

    # Load pre-trained EfficientNet_B3
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

    # Freeze all parameters in the model initially
    for param in model.parameters():
        param.requires_grad = True


    num_ftrs = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    model = model.to(config.DEVICE)

    print(f"Model: EfficientNet_B3 (full fine-tuning)")
    print(f"Output classes: {num_classes}")
    print(f"Device: {config.DEVICE}")

    return model
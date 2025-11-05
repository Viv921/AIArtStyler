# model.py

import torch
import torch.nn as nn
from torchvision import models
import config


def create_model(num_classes=config.NUM_CLASSES):
    """
    Loads a pre-trained ResNet50 model and adapts it for fine-tuning
    the last convolutional block (layer4) and the new FC layer.
    """

    # 1. Load pre-trained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # 2. Freeze all parameters in the model initially
    for param in model.parameters():
        param.requires_grad = False

    # 3. Unfreeze the parameters of the final convolutional block (
    for param in model.layer4.parameters():
        param.requires_grad = True

    # 4. Replace the final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(config.DEVICE)

    print(f"Model: ResNet50 (fine-tuning 'layer4' and 'fc' layer)")
    print(f"Output classes: {num_classes}")
    print(f"Device: {config.DEVICE}")

    return model
# dataset.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import config


def get_dataloaders():
    """
    Loads the WikiArt dataset, applies transformations, splits it,
    and returns DataLoaders for train, val, and test sets.
    """
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomResizedCrop(config.CROP_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)
    ])

    # Validation/Test transform does not include augmentation
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.CenterCrop(config.CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)
    ])

    # Load the full dataset
    try:
        full_dataset = datasets.ImageFolder(
            root=config.DATA_DIR,
            transform=train_transform
        )
    except FileNotFoundError:
        print(f"Error: Data directory not found at {config.DATA_DIR}")
        print("Please download the WikiArt dataset and place it in that directory.")
        return None, None, None, None

    # Get class names
    class_names = full_dataset.classes
    print(f"Found {len(class_names)} classes:")
    print(class_names)

    # Define split sizes
    total_size = len(full_dataset)
    train_size = int(config.TRAIN_SPLIT * total_size)
    val_size = int(config.VAL_SPLIT * total_size)
    test_size = total_size - train_size - val_size

    # Split the dataset
    # We use a fixed generator for reproducible splits
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # IMPORTANT: Apply the correct transform to val/test splits
    # The 'full_dataset' was initialized with 'train_transform'.
    # We need to override the .dataset attribute for val and test splits
    # to point to a new ImageFolder instance with the 'val_test_transform'.

    # Create a new dataset instance for validation and testing
    val_test_dataset = datasets.ImageFolder(
        root=config.DATA_DIR,
        transform=val_test_transform
    )

    # Manually assign the correct dataset (with the right transform)
    # to the val_dataset and test_dataset subsets.
    val_dataset.dataset = val_test_dataset
    test_dataset.dataset = val_test_dataset

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, class_names
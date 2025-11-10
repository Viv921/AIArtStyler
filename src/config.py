# config.py

import torch

# -- Paths --
DATA_DIR = "../dataset/data"
MODEL_PATH = "../models/art_classifier.pth"

# -- Compute --
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# -- Dataset --
# ImageNet stats for normalization
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 320
CROP_SIZE = 300

# -- Data Splitting --

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15


# -- Model & Training --
NUM_CLASSES = 13
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4

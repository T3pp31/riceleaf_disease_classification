
import torch
import os

# Data paths
DATA_DIR = '/root/github/riceleaf_disease_classification/data'
OUTPUT_DIR = '/root/github/riceleaf_disease_classification/train_model/output'
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Model parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (224, 224)
NUM_CLASSES = 4
CLASS_NAMES = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']
MODEL_NAME = 'efficientnet_b0'  # convnext_tiny, vit_small_patch16_224, swin_tiny_patch4_window7_224
DROP_RATE = 0.2
DROP_PATH_RATE = 0.2
FREEZE_BASE = True

# Training hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-4
LR_SCHEDULER = 'cosine'  # cosine, step, exponential
WARMUP_EPOCHS = 3
MIN_LR = 1e-6

# Augmentation
USE_TIMM_AUGMENTATION = True
USE_MIXUP = True

# W&B
USE_WANDB = True
WANDB_PROJECT = 'riceleaf-disease-classification'

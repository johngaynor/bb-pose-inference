"""
Configuration settings for the pose classifier training.
"""
import torch

# --- Directories ---
TRAIN_DIR = r"images/sorted/training"
VAL_DIR = r"images/sorted/validation"

# --- Training hyperparameters ---
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

# --- Device configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- File paths ---
MODEL_SAVE_PATH = "pose_classifier.pth"
CLASS_MAPPING_PATH = "class_mapping.json"
COMPLETE_CHECKPOINT_PATH = "pose_classifier_complete.pth"

# --- Data loader settings ---
NUM_WORKERS = 0

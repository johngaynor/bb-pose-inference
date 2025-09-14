"""
Main script for data loading/preproccessing, model creation, training, and saving.
"""
import torch
from data import download_images_from_s3, sort_images_to_datasets, create_data_loaders
from model import create_model, save_model, train_model
from config import EPOCHS, LEARNING_RATE, PATIENCE

# Download images
print(f"---------- STARTING PROCESS ----------")
download_images_from_s3()

# Sort images into training and validation datasets
print(f"\n---------- SORTING IMAGES ----------")
stats = sort_images_to_datasets()
print(stats)

# Create model
print(f"\n---------- CREATING DATA LOADERS ----------")
train_loader, val_loader, class_names, num_classes = create_data_loaders()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(num_classes=num_classes, device=device)

# Train the model
print(f"\n---------- TRAINING MODEL ----------")
trained_model, training_history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    num_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    patience=PATIENCE
)

# Save the trained model
print(f"\n---------- SAVING MODEL ----------")

# Create comprehensive class mapping
class_to_idx = train_loader.dataset.class_to_idx
class_mapping = {
    "class_to_idx": class_to_idx,
    "idx_to_class": {str(idx): class_name for class_name, idx in class_to_idx.items()},
    "classes": list(class_to_idx.keys())
}

save_model(
    model=trained_model,
    class_mapping=class_mapping,
    num_classes=num_classes,
    model_path="pose_classifier.pth",
    complete_checkpoint_path="pose_classifier_complete.pth"
)



print(f"---------- ENDING PROCESS ----------")
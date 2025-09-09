"""
Data processing and loading utilities for pose classification.

This module handles:
1. Downloading images from S3 bucket
2. Loading and parsing the labels from config/db_labels.json
3. Organizing images from ./images into training/validation directories
4. Creating proper directory structure for PyTorch ImageFolder
5. Creating data loaders for training
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import shutil
import random
from pathlib import Path
import boto3
from tqdm import tqdm
from config.config import TRAIN_DIR, VAL_DIR, BATCH_SIZE, NUM_WORKERS, CLASS_MAPPING_PATH

def download_images_from_s3(bucket_name="checkin-poses", local_dir="images"):
    """
    Download images from S3 bucket using boto3.
    
    Args:
        bucket_name (str): Name of the S3 bucket
        local_dir (str): Local directory to download images to
    """
    print(f"Checking for existing images in {local_dir}...")
    
    # Create local directory if it doesn't exist
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if images directory has content
    local_path = Path(local_dir)
    existing_files = list(local_path.glob("*"))
    
    if existing_files:
        print(f"Found {len(existing_files)} existing files in {local_dir}")
        while True:
            response = input("❔ Do you want to use existing images? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                print("Using existing images, skipping download.")
                return True
            elif response in ['n', 'no']:
                print("Clearing existing files before downloading...")
                for file_path in existing_files:
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"Warning: Could not remove {file_path}: {e}")
                print("Directory cleared. Proceeding with download...")
                break
            else:
                print("Please enter 'y' or 'n'")
    
    # Initialize S3 client
    try:
        s3_client = boto3.client('s3')
    except Exception as e:
        print(f"ERROR: Failed to create S3 client: {e}")
        print("Make sure AWS credentials are configured:")
        print("  - Run 'aws configure' to set up credentials")
        print("  - Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        print("  - Or ensure IAM role is configured if running on EC2")
        return False
    
    try:
        # List all objects in the bucket
        print("Fetching list of objects from S3...")
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)
        
        # Collect all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_objects = []
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if any(key.lower().endswith(ext) for ext in image_extensions):
                        image_objects.append(key)
        
        print(f"Found {len(image_objects)} images in S3 bucket")
        
        if not image_objects:
            print("No images found in S3 bucket!")
            return False
        
        # Download images with progress bar
        downloaded_count = 0
        skipped_count = 0
        
        for key in tqdm(image_objects, desc="Downloading images"):
            local_path = Path(local_dir) / key
            
            # Skip if file already exists and has the same size
            if local_path.exists():
                try:
                    # Get S3 object info
                    response = s3_client.head_object(Bucket=bucket_name, Key=key)
                    s3_size = response['ContentLength']
                    local_size = local_path.stat().st_size
                    
                    if s3_size == local_size:
                        skipped_count += 1
                        continue
                except Exception:
                    pass  # Re-download if we can't compare sizes
            
            # Create parent directory if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download the file
            try:
                s3_client.download_file(bucket_name, key, str(local_path))
                downloaded_count += 1
            except Exception as e:
                print(f"Failed to download {key}: {e}")
        
        print(f"Download complete! Downloaded: {downloaded_count}, Skipped: {skipped_count}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to download images from S3: {e}")
        return False


def sort_images_to_datasets(images_dir="images", labels_file="config/db_labels.json", poses_file="config/poses.json", validation_split=0.2, random_seed=42):
    """
    Sort images from the downloaded images directory into training and validation datasets
    based on pose labels from the database.
    
    Args:
        images_dir (str): Directory containing downloaded images
        labels_file (str): Path to the JSON file containing pose labels
        poses_file (str): Path to the JSON file containing pose definitions
        validation_split (float): Proportion of data to use for validation (0.0 to 1.0)
        random_seed (int): Random seed for reproducible splits
    
    Returns:
        dict: Statistics about the sorting process
    """
    print("🔄 Sorting images into training and validation datasets...")
    
    # Set random seed for reproducible splits
    random.seed(random_seed)
    
    # Load pose definitions from JSON file
    try:
        with open(poses_file, 'r') as f:
            poses_data = json.load(f)
        pose_id_to_name = {pose['id']: pose['name'] for pose in poses_data}
        print(f"✅ Loaded {len(poses_data)} pose definitions from {poses_file}")
    except Exception as e:
        print(f"❌ ERROR: Could not load poses file {poses_file}: {e}")
        return None
    
    # Load labels from JSON file
    try:
        with open(labels_file, 'r') as f:
            labels_data = json.load(f)
        print(f"✅ Loaded {len(labels_data)} labels from {labels_file}")
    except Exception as e:
        print(f"❌ ERROR: Could not load labels file {labels_file}: {e}")
        return None
    
    # Create a mapping from s3Filename to poseId
    filename_to_pose = {}
    for record in labels_data:
        filename_to_pose[record['s3Filename']] = record['poseId']
    
    # Get list of all images in the images directory
    images_path = Path(images_dir)
    if not images_path.exists():
        print(f"❌ ERROR: Images directory {images_dir} does not exist")
        return None
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Use a set to avoid duplicates from case-insensitive filesystem
    image_files_set = set()
    for ext in image_extensions:
        image_files_set.update(images_path.glob(f"*{ext}"))
        image_files_set.update(images_path.glob(f"*{ext.upper()}"))
    
    image_files = list(image_files_set)
    print(f"📁 Found {len(image_files)} image files in {images_dir}")
    
    # Group images by pose ID
    images_by_pose = {}
    labeled_count = 0
    unlabeled_files = []
    
    for image_file in image_files:
        filename = image_file.name
        if filename in filename_to_pose:
            pose_id = filename_to_pose[filename]
            if pose_id not in images_by_pose:
                images_by_pose[pose_id] = []
            images_by_pose[pose_id].append(image_file)
            labeled_count += 1
        else:
            unlabeled_files.append(filename)
    
    print(f"📊 Labeled images: {labeled_count}, Unlabeled images: {len(unlabeled_files)}")
    
    if unlabeled_files:
        print(f"⚠️  Warning: {len(unlabeled_files)} images have no labels and will be skipped")
        if len(unlabeled_files) <= 10:
            print("Unlabeled files:", unlabeled_files)
        else:
            print(f"First 10 unlabeled files: {unlabeled_files[:10]}")
    
    # Create directory structure
    train_dir = Path(TRAIN_DIR)
    val_dir = Path(VAL_DIR)
    
    # Check if sorted directories already exist
    if train_dir.exists() and val_dir.exists() and any(train_dir.iterdir()) and any(val_dir.iterdir()):
        print(f"📁 Found existing sorted directories:")
        print(f"   Training: {train_dir}")
        print(f"   Validation: {val_dir}")
        
        while True:
            response = input("❔ Do you want to use existing sorted directories? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                print("✅ Using existing sorted directories, skipping image sorting.")
                
                # Return basic stats from existing directories
                existing_stats = {
                    'total_images_processed': 0,
                    'train_count': 0,
                    'val_count': 0,
                    'poses': {},
                    'unlabeled_count': len(unlabeled_files),
                    'using_existing': True
                }
                
                # Count existing images
                for pose_dir in train_dir.iterdir():
                    if pose_dir.is_dir():
                        pose_id = pose_dir.name
                        train_count = len(list(pose_dir.glob("*")))
                        val_count = len(list((val_dir / pose_id).glob("*"))) if (val_dir / pose_id).exists() else 0
                        
                        pose_name = pose_id_to_name.get(int(pose_id) if pose_id.isdigit() else pose_id, f"Unknown Pose {pose_id}")
                        existing_stats['poses'][pose_id] = {
                            'pose_id': int(pose_id) if pose_id.isdigit() else pose_id,
                            'pose_name': pose_name,
                            'total': train_count + val_count,
                            'train': train_count,
                            'val': val_count
                        }
                        existing_stats['train_count'] += train_count
                        existing_stats['val_count'] += val_count
                        existing_stats['total_images_processed'] += train_count + val_count
                
                print(f"\n📊 Existing Dataset Summary:")
                print(f"🎯 Training images: {existing_stats['train_count']}")
                print(f"🎯 Validation images: {existing_stats['val_count']}")
                print(f"🔢 Total images: {existing_stats['total_images_processed']}")
                
                return existing_stats
                
            elif response in ['n', 'no']:
                print("🧹 Clearing existing directories and creating fresh sort...")
                break
            else:
                print("Please enter 'y' or 'n'")
    
    # Remove existing sorted directories if they exist
    if train_dir.exists():
        print(f"🧹 Cleaning existing training directory: {train_dir}")
        shutil.rmtree(train_dir)
    if val_dir.exists():
        print(f"🧹 Cleaning existing validation directory: {val_dir}")
        shutil.rmtree(val_dir)
    
    # Create new directory structure
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Statistics tracking
    stats = {
        'total_images_processed': 0,
        'train_count': 0,
        'val_count': 0,
        'poses': {},
        'unlabeled_count': len(unlabeled_files)
    }
    
    # Process each pose
    for pose_id, pose_images in images_by_pose.items():
        pose_name = pose_id_to_name.get(pose_id, f"Unknown Pose {pose_id}")
        pose_dir_name = str(pose_id)  # Use pose ID as directory name
        print(f"📂 Processing pose {pose_id} ({pose_name}): {len(pose_images)} images")
        
        # Create pose directories using pose ID
        train_pose_dir = train_dir / pose_dir_name
        val_pose_dir = val_dir / pose_dir_name
        train_pose_dir.mkdir(exist_ok=True)
        val_pose_dir.mkdir(exist_ok=True)
        
        # Shuffle images for this pose
        pose_images_shuffled = pose_images.copy()
        random.shuffle(pose_images_shuffled)
        
        # Split into train/val
        val_count = int(len(pose_images_shuffled) * validation_split)
        train_count = len(pose_images_shuffled) - val_count
        
        train_images = pose_images_shuffled[:train_count]
        val_images = pose_images_shuffled[train_count:]
        
        # Copy training images
        for img_file in train_images:
            dest_path = train_pose_dir / img_file.name
            shutil.copy2(img_file, dest_path)
        
        # Copy validation images
        for img_file in val_images:
            dest_path = val_pose_dir / img_file.name
            shutil.copy2(img_file, dest_path)
        
        # Update statistics
        stats['poses'][pose_dir_name] = {
            'pose_id': pose_id,
            'pose_name': pose_name,
            'total': len(pose_images_shuffled),
            'train': len(train_images),
            'val': len(val_images)
        }
        stats['train_count'] += len(train_images)
        stats['val_count'] += len(val_images)
        stats['total_images_processed'] += len(pose_images_shuffled)
        
        print(f"  ✅ Pose {pose_id} ({pose_name}): {len(train_images)} training, {len(val_images)} validation")
    
    # Print final statistics
    print("\n📊 Dataset Sorting Complete!")
    print(f"📁 Training directory: {train_dir}")
    print(f"📁 Validation directory: {val_dir}")
    print(f"🔢 Total images processed: {stats['total_images_processed']}")
    print(f"🎯 Training images: {stats['train_count']}")
    print(f"🎯 Validation images: {stats['val_count']}")
    print(f"⚠️  Unlabeled images skipped: {stats['unlabeled_count']}")
    
    print("\n📋 Breakdown by pose:")
    for pose_dir, counts in stats['poses'].items():
        pose_name = counts['pose_name']
        print(f"  Pose {pose_dir} ({pose_name}): {counts['total']} total ({counts['train']} train, {counts['val']} val)")
    
    return stats


def create_data_loaders(train_dir=TRAIN_DIR, val_dir=VAL_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, input_size=224):
    """
    Create PyTorch data loaders for training and validation datasets.
    
    Args:
        train_dir (str): Path to training data directory
        val_dir (str): Path to validation data directory
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        input_size (int): Target image size for resizing
    
    Returns:
        tuple: (train_loader, val_loader, class_names, num_classes)
    """
    print("🔄 Creating data loaders...")
    
    # Check if directories exist
    train_path = Path(train_dir)
    val_path = Path(val_dir)
    
    if not train_path.exists():
        print(f"❌ ERROR: Training directory {train_dir} does not exist")
        print("Make sure to run sort_images_to_datasets() first")
        return None, None, None, 0
    
    if not val_path.exists():
        print(f"❌ ERROR: Validation directory {val_dir} does not exist")
        print("Make sure to run sort_images_to_datasets() first")
        return None, None, None, 0
    
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    try:
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
        
        print(f"✅ Training dataset: {len(train_dataset)} images")
        print(f"✅ Validation dataset: {len(val_dataset)} images")
        
        # Get class names and create mapping
        class_names = train_dataset.classes
        num_classes = len(class_names)
        
        print(f"📊 Found {num_classes} classes: {class_names}")
        
        # Save class mapping to JSON file with complete mapping information
        class_to_idx = train_dataset.class_to_idx
        
        # Create comprehensive class mapping
        class_mapping = {
            "class_to_idx": class_to_idx,
            "idx_to_class": {str(idx): class_name for class_name, idx in class_to_idx.items()},
            "classes": list(class_to_idx.keys())
        }
        
        with open(CLASS_MAPPING_PATH, 'w') as f:
            json.dump(class_mapping, f, indent=2)
        print(f"💾 Class mapping saved to {CLASS_MAPPING_PATH}")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to create datasets: {e}")
        return None, None, None, 0
    
    # Create data loaders
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        print(f"✅ Data loaders created successfully")
        print(f"📦 Training batches: {len(train_loader)}")
        print(f"📦 Validation batches: {len(val_loader)}")
        
        return train_loader, val_loader, class_names, num_classes
        
    except Exception as e:
        print(f"❌ ERROR: Failed to create data loaders: {e}")
        return None, None, None, 0


def get_dataset_stats(train_dir=TRAIN_DIR, val_dir=VAL_DIR):
    """
    Get statistics about the current training and validation datasets.
    
    Args:
        train_dir (str): Path to training data directory
        val_dir (str): Path to validation data directory
    
    Returns:
        dict: Dataset statistics
    """
    print("📊 Analyzing dataset statistics...")
    
    stats = {
        'training': {},
        'validation': {},
        'total_training': 0,
        'total_validation': 0,
        'classes': []
    }
    
    # Analyze training directory
    train_path = Path(train_dir)
    if train_path.exists():
        for class_dir in train_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                image_count = len(list(class_dir.glob("*")))
                stats['training'][class_name] = image_count
                stats['total_training'] += image_count
                if class_name not in stats['classes']:
                    stats['classes'].append(class_name)
    
    # Analyze validation directory
    val_path = Path(val_dir)
    if val_path.exists():
        for class_dir in val_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                image_count = len(list(class_dir.glob("*")))
                stats['validation'][class_name] = image_count
                stats['total_validation'] += image_count
                if class_name not in stats['classes']:
                    stats['classes'].append(class_name)
    
    # Print statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"🎯 Total training images: {stats['total_training']}")
    print(f"🎯 Total validation images: {stats['total_validation']}")
    print(f"🏷️  Number of classes: {len(stats['classes'])}")
    
    print(f"\n📋 Class breakdown:")
    for class_name in sorted(stats['classes']):
        train_count = stats['training'].get(class_name, 0)
        val_count = stats['validation'].get(class_name, 0)
        total = train_count + val_count
        print(f"  {class_name}: {total} total ({train_count} train, {val_count} val)")
    
    return stats
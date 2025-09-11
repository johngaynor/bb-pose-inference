"""
S3 Image Cleansing Script
Identifies images in S3 buckets that are not present in the database labels
and offers to delete them to keep buckets clean.
"""

import boto3
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

def load_db_labels(labels_file="config/db_labels.json"):
    """
    Load database labels and extract S3 filenames
    
    Args:
        labels_file (str): Path to the database labels JSON file
    
    Returns:
        set: Set of S3 filenames that are labeled in the database
    """
    try:
        with open(labels_file, 'r') as f:
            labels_data = json.load(f)
        
        # Extract S3 filenames from the labels
        labeled_filenames = set()
        for item in labels_data:
            if 's3Filename' in item:
                labeled_filenames.add(item['s3Filename'])
        
        return labeled_filenames
    
    except FileNotFoundError:
        print(f"❌ Labels file not found: {labels_file}")
        return set()
    except Exception as e:
        print(f"❌ Error loading labels: {e}")
        return set()

def get_s3_images(bucket_name, s3_client):
    """
    Get all image files from an S3 bucket
    
    Args:
        bucket_name (str): Name of the S3 bucket
        s3_client: Boto3 S3 client
    
    Returns:
        list: List of image filenames in the bucket
    """
    try:
        print(f"📦 Scanning bucket: {bucket_name}")
        
        # List all objects in the bucket
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)
        
        # Collect all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if any(key.lower().endswith(ext) for ext in image_extensions):
                        # Extract just the filename (remove path if present)
                        filename = Path(key).name
                        image_files.append({
                            'key': key,
                            'filename': filename,
                            'size': obj['Size']
                        })
        
        return image_files
    
    except Exception as e:
        print(f"❌ Error scanning bucket {bucket_name}: {e}")
        return []

def find_unlabeled_images(bucket_names, labeled_filenames, s3_client):
    """
    Find images in S3 buckets that are not in the database labels
    
    Args:
        bucket_names (list): List of S3 bucket names to check
        labeled_filenames (set): Set of filenames that are labeled
        s3_client: Boto3 S3 client
    
    Returns:
        dict: Dictionary mapping bucket names to lists of unlabeled images
    """
    unlabeled_images = defaultdict(list)
    bucket_stats = {}
    
    for bucket_name in bucket_names:
        print(f"\n🔍 Analyzing bucket: {bucket_name}")
        print("-" * 50)
        
        # Get all images in this bucket
        bucket_images = get_s3_images(bucket_name, s3_client)
        
        if not bucket_images:
            print(f"   No images found in {bucket_name}")
            bucket_stats[bucket_name] = {'total': 0, 'unlabeled': 0, 'labeled': 0}
            continue
        
        # Check which images are not labeled
        labeled_count = 0
        unlabeled_count = 0
        
        for image_info in bucket_images:
            filename = image_info['filename']
            
            if filename in labeled_filenames:
                labeled_count += 1
            else:
                unlabeled_count += 1
                unlabeled_images[bucket_name].append(image_info)
        
        bucket_stats[bucket_name] = {
            'total': len(bucket_images),
            'labeled': labeled_count,
            'unlabeled': unlabeled_count
        }
        
        print(f"   📊 Total images: {len(bucket_images)}")
        print(f"   ✅ Labeled: {labeled_count}")
        print(f"   ❌ Unlabeled: {unlabeled_count}")
        
        if unlabeled_count > 0:
            print(f"   📋 Sample unlabeled files:")
            for i, img in enumerate(unlabeled_images[bucket_name][:5]):
                size_mb = img['size'] / (1024 * 1024)
                print(f"      - {img['filename']} ({size_mb:.1f}MB)")
            if unlabeled_count > 5:
                print(f"      ... and {unlabeled_count - 5} more")
    
    return unlabeled_images, bucket_stats

def delete_unlabeled_images(unlabeled_images, s3_client):
    """
    Delete unlabeled images from S3 buckets
    
    Args:
        unlabeled_images (dict): Dictionary of unlabeled images by bucket
        s3_client: Boto3 S3 client
    """
    total_deleted = 0
    total_failed = 0
    
    for bucket_name, images in unlabeled_images.items():
        if not images:
            continue
        
        print(f"\n🗑️  Deleting from bucket: {bucket_name}")
        bucket_deleted = 0
        bucket_failed = 0
        
        # Delete images with progress bar
        for image_info in tqdm(images, desc=f"Deleting from {bucket_name}", leave=False):
            try:
                s3_client.delete_object(Bucket=bucket_name, Key=image_info['key'])
                bucket_deleted += 1
            except Exception as e:
                print(f"   ❌ Failed to delete {image_info['filename']}: {e}")
                bucket_failed += 1
        
        print(f"   ✅ Deleted: {bucket_deleted}")
        if bucket_failed > 0:
            print(f"   ❌ Failed: {bucket_failed}")
        
        total_deleted += bucket_deleted
        total_failed += bucket_failed
    
    print(f"\n🎉 Deletion complete!")
    print(f"   Total deleted: {total_deleted}")
    if total_failed > 0:
        print(f"   Total failed: {total_failed}")

def main():
    """Main cleansing function"""
    
    # Configuration
    BUCKET_NAMES = ["checkin-poses", "checkin-photos"]  # Add your bucket names here
    LABELS_FILE = "config/db_labels.json"
    
    print("🧹 S3 Image Cleansing Tool")
    print("=" * 50)
    print(f"📋 Checking buckets: {', '.join(BUCKET_NAMES)}")
    print(f"🏷️  Using labels from: {LABELS_FILE}")
    
    # Initialize S3 client
    try:
        s3_client = boto3.client('s3')
        print("✅ S3 client initialized successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to create S3 client: {e}")
        print("Make sure AWS credentials are configured:")
        print("  - Run 'aws configure' to set up credentials")
        print("  - Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        return
    
    # Load database labels
    print(f"\n📖 Loading database labels...")
    labeled_filenames = load_db_labels(LABELS_FILE)
    
    if not labeled_filenames:
        print("❌ No labeled filenames found. Cannot proceed.")
        return
    
    print(f"   Found {len(labeled_filenames)} labeled images in database")
    
    # Find unlabeled images
    print(f"\n🔍 Scanning S3 buckets for unlabeled images...")
    unlabeled_images, bucket_stats = find_unlabeled_images(BUCKET_NAMES, labeled_filenames, s3_client)
    
    # Display summary
    print(f"\n📊 CLEANSING SUMMARY")
    print("=" * 60)
    
    total_unlabeled = 0
    total_images = 0
    
    for bucket_name in BUCKET_NAMES:
        stats = bucket_stats.get(bucket_name, {'total': 0, 'labeled': 0, 'unlabeled': 0})
        total_images += stats['total']
        total_unlabeled += stats['unlabeled']
        
        print(f"📦 {bucket_name}:")
        print(f"   Total: {stats['total']}, Labeled: {stats['labeled']}, Unlabeled: {stats['unlabeled']}")
    
    print(f"\n🎯 OVERALL TOTALS:")
    print(f"   📊 Total images across all buckets: {total_images}")
    print(f"   ✅ Total labeled images: {total_images - total_unlabeled}")
    print(f"   ❌ Total unlabeled images: {total_unlabeled}")
    
    if total_unlabeled == 0:
        print("\n🎉 All images are properly labeled! No cleanup needed.")
        return
    
    # Calculate storage savings
    total_size_mb = 0
    for bucket_name, images in unlabeled_images.items():
        for img in images:
            total_size_mb += img['size'] / (1024 * 1024)
    
    print(f"   💾 Storage to be freed: {total_size_mb:.1f} MB")
    
    # Ask user for confirmation
    print(f"\n⚠️  WARNING: This will permanently delete {total_unlabeled} unlabeled images!")
    print("   Make sure you have backups if needed.")
    
    while True:
        response = input(f"\n❔ Do you want to delete these {total_unlabeled} unlabeled images? (y/n): ").strip().lower()
        
        if response in ['y', 'yes']:
            print(f"\n🗑️  Starting deletion process...")
            delete_unlabeled_images(unlabeled_images, s3_client)
            break
        elif response in ['n', 'no']:
            print(f"\n✋ Deletion cancelled. No images were deleted.")
            break
        else:
            print("Please enter 'y' or 'n'")
    
    print(f"\n🏁 Cleansing process complete!")

if __name__ == "__main__":
    main()

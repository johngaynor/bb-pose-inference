# 🤸‍♀️ PhysiQ Pose Classifier

A deep learning-based pose classification system that automatically identifies fitness poses from images using PyTorch and ResNet50 architecture. The system downloads images from AWS S3, preprocesses them, and trains a neural network to classify different fitness poses.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Training](#training)
- [Inference](#inference)
- [AWS S3 Integration](#aws-s3-integration)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **🏗️ Multi-Bucket S3 Integration**: Download training images from multiple AWS S3 buckets
- **🧠 Transfer Learning**: Uses pre-trained ResNet50 with fine-tuning for pose classification
- **📊 Enhanced Progress Monitoring**: Real-time training progress with epoch-level and batch-level monitoring
- **🔄 Automatic Dataset Splitting**: Intelligent train/validation split with pose-based organization
- **📈 Early Stopping**: Prevents overfitting with configurable patience
- **🎯 Comprehensive Inference**: Test individual images with confidence scores and probability distributions
- **💾 Model Checkpointing**: Save and load complete model states with class mappings
- **📋 Flexible Configuration**: JSON-based pose definitions and class mappings
- **🧹 S3 Cleansing Tool**: Identify and remove unlabeled images from S3 buckets to optimize storage

## 🏛️ Architecture

- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Input Size**: 224x224 RGB images
- **Output**: Multi-class classification (8 pose classes)
- **Framework**: PyTorch with torchvision
- **Optimization**: Adam optimizer with CrossEntropyLoss

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended)
- AWS credentials configured

### Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/johngaynor/bb-pose-classifier.git
   cd bb-pose-classifier
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure AWS credentials**:
   ```bash
   aws configure
   # Or set environment variables:
   # AWS_ACCESS_KEY_ID=your_access_key
   # AWS_SECRET_ACCESS_KEY=your_secret_key
   ```

## ⚙️ Configuration

### Pose Classes

The system currently supports 8 fitness pose classes:

| Class ID | Pose Name           |
| -------- | ------------------- |
| 1        | Front Lat Spread    |
| 2        | Front Double Biceps |
| 5        | Back Double Biceps  |
| 7        | Back Lat Spread     |
| 11       | Abs & Thighs        |
| 14       | Side Chest          |
| 16       | Side Triceps        |

### Configuration Files

- **`config.py`**: Training hyperparameters and paths
- **`mappings/poses.json`**: Pose definitions and metadata
- **`mappings/db_labels.json`**: Database labels for image-pose mapping
- **`mappings/class_mapping.json`**: Generated class mapping (created during training)

## 🎯 Usage

### Complete Training Pipeline

Run the full pipeline from data download to model training:

```bash
python main.py
```

This will:

1. Download images from S3 bucket(s)
2. Sort images into training/validation datasets
3. Create PyTorch data loaders
4. Train the ResNet50 model
5. Save the trained model and class mappings

### Multi-Bucket Download

To download from multiple S3 buckets, modify the call in `main.py`:

```python
# Single bucket (default)
download_images_from_s3()

# Multiple buckets
download_images_from_s3([
    "checkin-poses",
    "workout-images",
    "fitness-photos"
])
```

### Individual Image Inference

Test the trained model on a single image:

```bash
python misc/test.py
```

Update the `IMAGE_PATH` variable in `misc/test.py` to point to your test image:

```python
IMAGE_PATH = r"test_images/your_image.jpg"
```

### S3 Bucket Cleansing

Clean up your S3 buckets by removing unlabeled images:

```bash
python misc/cleanse.py
```

This tool will:
1. Scan your S3 buckets for all images
2. Compare against database labels to find unlabeled images
3. Show detailed statistics and storage savings
4. Optionally delete unlabeled images with user confirmation

## 📁 Project Structure

```
bb-pose-classifier/
├── mappings/
│   ├── poses.json             # Pose definitions
│   ├── db_labels.json         # Database labels
│   └── class_mapping.json     # Generated class mappings
├── misc/
│   ├── test.py                # Inference script
│   └── cleanse.py             # S3 bucket cleaning tool
├── images/                    # Downloaded images
├── test_images/              # Test images for inference
├── config.py                 # Training configuration
├── main.py                   # Main training pipeline
├── data.py                   # Data loading and preprocessing
├── model.py                  # Model architecture and training
├── requirements.txt          # Python dependencies
├── pose_classifier.pth       # Trained model (state dict only)
├── pose_classifier_complete.pth  # Complete checkpoint
└── README.md                 # This file
```

## 🧠 Model Details

### Architecture

- **Base**: ResNet50 pre-trained on ImageNet
- **Modifications**:
  - Final fully connected layer adapted for 8-class classification
  - Transfer learning with fine-tuning
  - Input normalization using ImageNet statistics

### Training Configuration

- **Batch Size**: 32
- **Learning Rate**: 0.0001 (Adam optimizer)
- **Epochs**: Up to 50 (with early stopping)
- **Early Stopping Patience**: 10 epochs
- **Validation Split**: 20%
- **Data Augmentation**: Resize to 224x224, normalization

### Class Mapping Format

```json
{
  "class_to_idx": {"1": 0, "2": 4, "5": 5, ...},
  "idx_to_class": {"0": "1", "4": "2", "5": "5", ...},
  "classes": ["1", "2", "5", "7", "11", "14", "16"]
}
```

## 🏋️‍♀️ Training

### Enhanced Progress Monitoring

The training process provides detailed progress tracking:

```
===========================================================================================
Epoch  Train Loss   Train Acc    Val Loss     Val Acc      Time     Status
===========================================================================================
1      2.0456       45.6%        1.8923       52.3%        12.4s    ✓ Best Model
2      1.7834       58.1%        1.6445       61.7%        11.8s    ✓ Best Model
3      1.5123       67.2%        1.7234       58.9%        12.1s    Wait 1/10
```

### Within-Epoch Progress

Real-time batch-level progress bars:

```
Epoch 1 [Train]: 100%|████████| 45/45 [00:08<00:00, 5.2it/s] Loss: 1.8456, Acc: 52.3%
Epoch 1 [Valid]: 100%|████████| 12/12 [00:01<00:00, 8.1it/s] Loss: 1.7234, Acc: 58.9%
```

### Model Checkpoints

Two types of model files are saved:

- **`pose_classifier.pth`**: Model state dict only (smaller file)
- **`pose_classifier_complete.pth`**: Complete checkpoint with optimizer state, class mappings, and metadata

## 🔮 Inference

### Test Script Features

The `test.py` script provides:

- **Visual Results**: Side-by-side original image and probability bar chart
- **Confidence Scores**: Probability distribution across all pose classes
- **Detailed Console Output**: Predicted pose, confidence percentage, and all class probabilities

### Sample Output

```
🧠 Pose Classification Inference
==================================================
🖥️  Using device: cuda
📋 Loading class mapping...
   Found 8 pose classes
🤖 Loading trained model...
   ✅ Model loaded successfully
🖼️  Processing image: test_image.jpg
   Original size: (1024, 768)
   Processed tensor shape: torch.Size([1, 3, 224, 224])

🎯 PREDICTION RESULTS
==============================
Predicted Pose: 2
Confidence: 87.3%

All Class Probabilities:
   Pose 1: 2.1%
🎯 Pose 2: 87.3%
   Pose 5: 8.4%
   Pose 7: 1.8%
   ...
```

## ☁️ AWS S3 Integration

### Multi-Bucket Support

Download images from multiple S3 buckets:

```python
download_images_from_s3([
    "primary-poses",
    "supplementary-poses",
    "validation-poses"
])
```

### Features

- **Smart Resuming**: Skips already downloaded files with size verification
- **Progress Tracking**: Individual progress bars for each bucket
- **Error Handling**: Continues with remaining buckets if one fails
- **Comprehensive Reporting**: Detailed statistics per bucket and overall totals

### AWS Configuration

Ensure AWS credentials are configured:

1. **AWS CLI**: `aws configure`
2. **Environment Variables**:
   ```bash
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   ```
3. **IAM Role**: If running on EC2

### S3 Bucket Cleansing

The included `misc/cleanse.py` tool helps maintain clean S3 buckets:

**Features:**
- **Unlabeled Image Detection**: Identifies images in S3 that aren't in your database labels
- **Multi-Bucket Analysis**: Scans multiple buckets simultaneously
- **Storage Optimization**: Shows potential storage savings before deletion
- **Safe Deletion**: Requires explicit user confirmation with detailed warnings
- **Progress Monitoring**: Visual feedback during scanning and deletion processes

**Sample Output:**
```
🧹 S3 Image Cleansing Tool
==================================================
📊 CLEANSING SUMMARY
============================================================
📦 checkin-poses:
   Total: 1456, Labeled: 1389, Unlabeled: 67
📦 checkin-photos:
   Total: 892, Labeled: 845, Unlabeled: 47

🎯 OVERALL TOTALS:
   📊 Total images across all buckets: 2,348
   ✅ Total labeled images: 2,234
   ❌ Total unlabeled images: 114
   💾 Storage to be freed: 127.4 MB
```

## 🔧 Customization

### Adding New Pose Classes

1. Update `mappings/poses.json` with new pose definitions
2. Add corresponding labels to `mappings/db_labels.json`
3. Ensure training images are organized in pose-specific directories
4. Retrain the model with updated class count

### Hyperparameter Tuning

Modify `config.py`:

```python
BATCH_SIZE = 64        # Increase if you have more GPU memory
LEARNING_RATE = 1e-3   # Adjust learning rate
EPOCHS = 100           # Increase maximum epochs
```

### Custom Data Sources

Replace the S3 download function with your own data loading logic while maintaining the expected directory structure.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch and torchvision teams for the excellent deep learning framework
- ResNet architecture from "Deep Residual Learning for Image Recognition"
- AWS S3 for scalable image storage
- The fitness and bodybuilding community for pose classification inspiration

---

**Built with ❤️ for fitness pose classification**

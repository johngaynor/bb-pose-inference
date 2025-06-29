# Pose Classification Model

A deep learning model for classifying bodybuilding poses using PyTorch and ResNet50.

## Overview

This project trains a convolutional neural network to classify different bodybuilding poses from images. The model can distinguish between poses like "Front Relaxed", "Back Double Biceps", "Side Chest", etc.

## Project Structure

```
poseAssignment/
├── train.ipynb              # Model training notebook
├── inference.ipynb          # Model inference notebook
├── testImage.ipynb          # Image testing notebook
├── pose_classifier.pth      # Trained model weights
├── class_mapping.json       # Class definitions and mappings
├── test_images/            # Sample test images
├── images/                 # Training data (not in repo)
│   └── sorted/
│       ├── training/       # Training images organized by class
│       └── validation/     # Validation images organized by class
└── README.md              # This file
```

## Setup and Usage

### 1. Format Data Using Data Script

Before training, you need to organize your images into the proper directory structure:

```
images/sorted/
├── training/
│   ├── pose_1/
│   ├── pose_2/
│   ├── pose_3/
│   └── ...
└── validation/
    ├── pose_1/
    ├── pose_2/
    ├── pose_3/
    └── ...
```

- Each pose should have its own folder named consistently (e.g., `pose_1`, `pose_2`, etc.)
- Split your data into training and validation sets (typically 80/20 split)
- Ensure image formats are supported (JPG, PNG)

### 2. Image Transformation Validation

Before training, verify that your images will be processed correctly:

- **Image Resolution**: Images are resized to 224x224 pixels (ResNet50 requirement)
- **Quality Check**: Ensure resizing doesn't lose critical pose details
- **Format**: Images are converted to RGB format
- **Normalization**: Images are normalized using ImageNet statistics

**Recommended image guidelines:**

- Minimum resolution: 224x224 (higher is better)
- Clear, well-lit photos
- Full body visible
- Minimal background distractions

You can test image transformations using the `testImage.ipynb` notebook.

### 3. Train the Model

Open and run `train.ipynb`:

```python
# Key training parameters
TRAIN_DIR = r"images/sorted/training"
VAL_DIR = r"images/sorted/validation"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
```

The training process will:

1. Load and preprocess your images
2. Create class mappings and save them to `class_mapping.json`
3. Train a ResNet50 model with transfer learning
4. Save the trained model to `pose_classifier.pth`
5. Create a complete checkpoint with metadata

**Training outputs:**

- `pose_classifier.pth` - Model weights
- `class_mapping.json` - Class definitions (critical for inference)
- `pose_classifier_complete.pth` - Complete checkpoint with metadata

### 4. Test with Inference

Use `inference.ipynb` to test your trained model:

```python
# Load the model and make predictions
pose, confidence, class_name = predict_pose("test_images/your_image.jpg")
print(f"Predicted Pose: {pose} (Confidence: {confidence:.2%})")
```

The inference notebook will:

1. Load the saved class mappings from `class_mapping.json`
2. Initialize the model with the correct architecture
3. Load the trained weights
4. Make predictions on new images

## Model Architecture

- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Transfer Learning**: Only the final classification layer is retrained
- **Input Size**: 224x224x3 RGB images
- **Output**: Softmax probabilities for each pose class

## Class Mappings

The model's class mappings are stored in `class_mapping.json` and include:

```json
{
  "class_to_idx": {
    "pose_1": 0,
    "pose_2": 1,
    "pose_3": 2,
    ...
  },
  "idx_to_class": {
    "0": "pose_1",
    "1": "pose_2",
    "2": "pose_3",
    ...
  },
  "classes": ["pose_1", "pose_2", "pose_3", ...]
}
```

**Human-readable pose names:**

- pose_1: "Front Relaxed"
- pose_2: "Back Relaxed"
- pose_3: "Quarter Turn (Left)"
- pose_4: "Quarter Turn (Right)"
- pose_5: "Back Double Biceps"
- pose_6: "Front Double Biceps"
- pose_7: "Front Lat Spread"
- pose_8: "Side Chest (Left)"
- pose_11: "Abs & Thighs"

## Dependencies

```bash
pip install torch torchvision tqdm
```

## Important Notes

### Order Independence

The model is designed to be **order-independent**:

- Class mappings are saved during training in `class_mapping.json`
- Inference loads these mappings instead of depending on folder order
- This ensures consistent predictions regardless of training environment

### Git Configuration

- Model weights (`*.pth`) are excluded from version control (large files)
- Class mappings (`class_mapping.json`) are included (essential for inference)
- Notebook outputs are stripped using `nbstripout` filter

### Troubleshooting

**Common Issues:**

1. **"class_mapping.json not found"**

   - Run the training notebook first to generate class mappings

2. **Model architecture mismatch**

   - Ensure the number of classes matches between training and inference

3. **Poor predictions**

   - Check image quality and preprocessing
   - Verify class balance in training data
   - Consider training for more epochs

4. **CUDA out of memory**
   - Reduce batch size
   - Use CPU instead: `DEVICE = torch.device('cpu')`

## Performance Tips

- **GPU Training**: Use CUDA if available for faster training
- **Data Augmentation**: Training includes random flips and rotations
- **Batch Size**: Adjust based on GPU memory (32 is a good default)
- **Epochs**: Start with 10, increase if underfitting

## License

This project is for educational/research purposes. Please ensure you have proper rights to use any training images.

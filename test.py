"""
Pose Classification Inference Script
Test a single image against the trained pose classification model
"""

import torch
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model import load_model
import os

def load_class_mapping(mapping_path="class_mapping.json"):
    """Load the class mapping from JSON file"""
    try:
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        return mapping
    except FileNotFoundError:
        print(f"❌ Class mapping file not found: {mapping_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading class mapping: {e}")
        return None

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image for model inference
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing (height, width)
    
    Returns:
        torch.Tensor: Preprocessed image tensor
        PIL.Image: Original image for display
    """
    try:
        # Load and convert image
        original_image = Image.open(image_path).convert("RGB")
        
        # Define preprocessing transforms (same as training)
        preprocess = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms
        processed_tensor = preprocess(original_image)
        
        # Add batch dimension
        processed_tensor = processed_tensor.unsqueeze(0)
        
        return processed_tensor, original_image
    
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None, None

def predict_pose(model, image_tensor, class_mapping, device):
    """
    Predict pose from preprocessed image tensor
    
    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed image tensor
        class_mapping: Class mapping dictionary
        device: torch.device
    
    Returns:
        tuple: (predicted_class, confidence, all_probabilities)
    """
    model.eval()
    
    with torch.no_grad():
        # Move tensor to device
        image_tensor = image_tensor.to(device)
        
        # Get model predictions
        outputs = model(image_tensor)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get predicted class and confidence
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Convert to Python values
        predicted_idx = predicted_idx.item()
        confidence = confidence.item()
        all_probs = probabilities.squeeze().cpu().numpy()
        
        # Get class name from mapping
        if "idx_to_class" in class_mapping:
            predicted_class = class_mapping["idx_to_class"][str(predicted_idx)]
        else:
            # Fallback for old format
            predicted_class = str(predicted_idx)
        
        return predicted_class, confidence, all_probs

def display_results(image, predicted_class, confidence, all_probabilities, class_mapping):
    """Display the image and prediction results"""
    
    # Create subplot layout
    plt.figure(figsize=(15, 6))
    
    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Input Image\nPredicted Pose: {predicted_class}\nConfidence: {confidence:.1%}", 
              fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Display probability distribution
    plt.subplot(1, 2, 2)
    
    # Get class names for x-axis
    if "classes" in class_mapping:
        class_names = class_mapping["classes"]
    elif "idx_to_class" in class_mapping:
        class_names = [class_mapping["idx_to_class"][str(i)] for i in range(len(all_probabilities))]
    else:
        class_names = [str(i) for i in range(len(all_probabilities))]
    
    # Create bar plot
    bars = plt.bar(class_names, all_probabilities * 100)
    
    # Highlight the predicted class
    predicted_idx = class_names.index(predicted_class)
    bars[predicted_idx].set_color('red')
    bars[predicted_idx].set_alpha(0.8)
    
    plt.title('Pose Classification Probabilities', fontweight='bold')
    plt.xlabel('Pose Class')
    plt.ylabel('Probability (%)')
    plt.ylim(0, 100)
    
    # Add percentage labels on bars
    for i, (bar, prob) in enumerate(zip(bars, all_probabilities)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    """Main inference function"""
    
    # ==================== CONFIGURATION ====================
    # 🔧 CHANGE THIS PATH to point to your test image
    IMAGE_PATH = r"test_images\tarining-pose-2-2.jpeg"  # Update this path!
    
    MODEL_PATH = "pose_classifier_complete.pth"
    CLASS_MAPPING_PATH = "class_mapping.json"
    
    print("🧠 Pose Classification Inference")
    print("=" * 50)
    
    # Check if image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Image not found: {IMAGE_PATH}")
        print("📁 Available test images:")
        test_dir = "test_images"
        if os.path.exists(test_dir):
            for img in os.listdir(test_dir)[:10]:  # Show first 10
                print(f"   - {img}")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load class mapping
    print("📋 Loading class mapping...")
    class_mapping = load_class_mapping(CLASS_MAPPING_PATH)
    if class_mapping is None:
        return
    
    print(f"   Found {len(class_mapping.get('classes', []))} pose classes")
    
    # Load model
    print("🤖 Loading trained model...")
    try:
        model, model_class_mapping = load_model(MODEL_PATH, device)
        print("   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return
    
    # Load and preprocess image
    print(f"🖼️  Processing image: {os.path.basename(IMAGE_PATH)}")
    image_tensor, original_image = preprocess_image(IMAGE_PATH)
    
    if image_tensor is None:
        return
    
    print(f"   Original size: {original_image.size}")
    print(f"   Processed tensor shape: {image_tensor.shape}")
    
    # Make prediction
    print("🔮 Making prediction...")
    predicted_class, confidence, all_probabilities = predict_pose(
        model, image_tensor, class_mapping, device
    )
    
    # Display results
    print("\n🎯 PREDICTION RESULTS")
    print("=" * 30)
    print(f"Predicted Pose: {predicted_class}")
    print(f"Confidence: {confidence:.1%}")
    print("\nAll Class Probabilities:")
    
    # Get class names for display
    if "classes" in class_mapping:
        class_names = class_mapping["classes"]
    else:
        class_names = [str(i) for i in range(len(all_probabilities))]
    
    for class_name, prob in zip(class_names, all_probabilities):
        marker = "🎯" if class_name == predicted_class else "  "
        print(f"{marker} Pose {class_name}: {prob:.1%}")
    
    # Display visual results
    print("\n📊 Displaying results...")
    display_results(original_image, predicted_class, confidence, 
                   all_probabilities, class_mapping)

if __name__ == "__main__":
    main()

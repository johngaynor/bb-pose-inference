"""
Model definition and utilities for pose classification.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
import time
from tqdm import tqdm


def create_model(num_classes, device):
    """
    Create and initialize a ResNet50 model for pose classification.
    
    Args:
        num_classes (int): Number of output classes
        device (torch.device): Device to move the model to
    
    Returns:
        torch.nn.Module: Initialized model
    """
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    return model


def save_model(model, class_mapping, num_classes, model_path, complete_checkpoint_path):
    """
    Save the trained model and complete checkpoint.
    
    Args:
        model (torch.nn.Module): Trained model
        class_mapping (dict): Class mapping information
        num_classes (int): Number of classes
        model_path (str): Path to save the model state dict
        complete_checkpoint_path (str): Path to save complete checkpoint
    """
    # Save model state dict
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save complete checkpoint with metadata
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_mapping': class_mapping,
        'model_config': {
            'num_classes': num_classes,
            'architecture': 'resnet50'
        }
    }
    torch.save(checkpoint, complete_checkpoint_path)
    print(f"Complete checkpoint saved to {complete_checkpoint_path}")


def load_model(checkpoint_path, device):
    """
    Load a complete model checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        device (torch.device): Device to load the model on
    
    Returns:
        tuple: (model, class_mapping)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    num_classes = checkpoint['model_config']['num_classes']
    model = create_model(num_classes, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint['class_mapping']


def train_model(model, train_loader, val_loader, device, num_epochs=50, learning_rate=0.001, patience=10):
    """
    Train the pose classification model.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run training on
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate for optimizer
        patience: Early stopping patience
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    print(f"🚀 Starting training on {device}")
    print(f"📊 Training samples: {len(train_loader.dataset)}")
    print(f"📊 Validation samples: {len(val_loader.dataset)}")
    print(f"🎯 Number of classes: {len(train_loader.dataset.classes)}")
    print(f"⚙️  Learning rate: {learning_rate}")
    print(f"🔄 Max epochs: {num_epochs}")
    
    # Loss function and optimizer (matching your notebook)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Early stopping variables
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    print("\n" + "="*95)
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'Time':<8} {'Status':<15}")
    print("="*95)
    
    def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch_num):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for training batches
        pbar = tqdm(dataloader, 
                   desc=f"Epoch {epoch_num+1} [Train]", 
                   leave=False, 
                   ncols=100,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Loss: {postfix}')
        
        batch_losses = []
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar with current loss and accuracy
            current_loss = running_loss / total
            current_acc = correct / total * 100
            batch_losses.append(loss.item())
            
            # Show running averages every 10 batches or on last batch
            if (batch_idx + 1) % 10 == 0 or batch_idx == len(dataloader) - 1:
                pbar.set_postfix_str(f"{current_loss:.4f}, Acc: {current_acc:.1f}%")
        
        pbar.close()
        return running_loss / total, correct / total

    def validate(model, dataloader, criterion, device, epoch_num):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for validation batches
        pbar = tqdm(dataloader, 
                   desc=f"Epoch {epoch_num+1} [Valid]", 
                   leave=False, 
                   ncols=100,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Loss: {postfix}')
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                # Update progress bar
                current_loss = running_loss / total
                current_acc = correct / total * 100
                
                if (batch_idx + 1) % 5 == 0 or batch_idx == len(dataloader) - 1:
                    pbar.set_postfix_str(f"{current_loss:.4f}, Acc: {current_acc:.1f}%")
        
        pbar.close()
        return running_loss / total, correct / total
    
    # Main training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Train and validate
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc * 100)  # Convert to percentage
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc * 100)  # Convert to percentage
        
        # Print epoch results with enhanced formatting
        epoch_time = time.time() - epoch_start_time
        
        # Determine status for this epoch
        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            status = "✓ Best Model"
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                status = "Early Stop"
            else:
                status = f"Wait {epochs_without_improvement}/{patience}"
        
        print(f"{epoch+1:<6} {train_loss:<12.4f} {train_acc*100:<11.1f}% {val_loss:<12.4f} {val_acc*100:<11.1f}% {epoch_time:<7.1f}s {status:<15}")
        
        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
            print(f"   No improvement for {patience} consecutive epochs")
            break
    
    print("="*95)
    print(f"🎉 Training completed!")
    print(f"🏆 Best validation accuracy: {best_val_acc*100:.2f}%")
    print("="*95)
    
    return model, history

#main.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for SimSurgSkill dataset processing and model training
with COCO format data and comprehensive evaluation
"""
import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from src.models import EfficientDetModel
from src.coco_data_loader import get_coco_data_loaders


def main():
    # Define paths - update these with your actual paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data/simsurgskill_2021_dataset")
    coco_dir = os.path.join(base_dir, "data/coco_format")
    results_dir = os.path.join(base_dir, "results")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*60)
    print("SIMSURGSKILL MODEL TRAINING")
    print("="*60)
    
    # Use already converted COCO format
    coco_paths = {
        'train_dir': os.path.join(coco_dir, 'train', 'images'),
        'val_dir': os.path.join(coco_dir, 'val', 'images'),
        'test_dir': os.path.join(coco_dir, 'test', 'images'),
        'train_ann': os.path.join(coco_dir, 'annotations', 'instances_train.json'),
        'val_ann': os.path.join(coco_dir, 'annotations', 'instances_val.json'),
        'test_ann': os.path.join(coco_dir, 'annotations', 'instances_test.json')
    }
    
    # Verify COCO files exist
    print("\nVerifying COCO format files...")
    for key, path in coco_paths.items():
        if os.path.exists(path):
            print(f"  ✅ {key}: {path}")
        else:
            print(f"  ❌ {key}: {path} NOT FOUND!")
            return
    
    # Get data loaders for COCO format data
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = get_coco_data_loaders(coco_paths, batch_size=4)
    print(f"  ✅ Training batches: {len(train_loader)}")
    print(f"  ✅ Validation batches: {len(val_loader)}")
    print(f"  ✅ Test batches: {len(test_loader)}")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Using device: {device}")
    
    model = EfficientDetModel(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("✅ Model initialized and ready for training")
    
    # Training loop
    num_epochs = 10
    train_losses = []
    val_losses = []
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    for epoch in range(num_epochs):
        print(f"\n📊 Epoch [{epoch+1}/{num_epochs}]")
        print("-" * 60)
        
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for i, (images, targets) in enumerate(train_loader):
            try:
                images = images.to(device)
                
                # Move targets to device
                for j in range(len(targets)):
                    targets[j]['boxes'] = targets[j]['boxes'].to(device)
                    targets[j]['labels'] = targets[j]['labels'].to(device)
                    targets[j]['image_id'] = targets[j]['image_id'].to(device)
                
                # Forward pass
                outputs = model(images, targets)
                
                # Extract loss from output dictionary
                if isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    print(f"  ⚠️  Warning: Unexpected output format at batch {i}")
                    print(f"       Output type: {type(outputs)}")
                    continue
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
                
                if (i + 1) % 10 == 0:
                    print(f'  Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                    
            except Exception as e:
                print(f"  ❌ Error in training batch {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)
        print(f'\n  📈 Training Loss: {avg_train_loss:.4f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for i, (images, targets) in enumerate(val_loader):
                try:
                    images = images.to(device)
                    
                    # Move targets to device
                    for j in range(len(targets)):
                        targets[j]['boxes'] = targets[j]['boxes'].to(device)
                        targets[j]['labels'] = targets[j]['labels'].to(device)
                        targets[j]['image_id'] = targets[j]['image_id'].to(device)
                    
                    # Forward pass
                    outputs = model(images, targets)
                    
                    # Extract loss
                    if isinstance(outputs, dict) and 'loss' in outputs:
                        loss = outputs['loss']
                        val_loss += loss.item()
                        num_val_batches += 1
                        
                except Exception as e:
                    print(f"  ❌ Error in validation batch {i}: {e}")
                    continue
        
        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0
        val_losses.append(avg_val_loss)
        print(f'  📉 Validation Loss: {avg_val_loss:.4f}')
    
    # Save the model
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    model_save_path = os.path.join(base_dir, 'models', 'model.pth')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f'✅ Model saved to: {model_save_path}')
    
    # Plot training curves
    print("\n" + "="*60)
    print("GENERATING TRAINING PLOTS")
    print("="*60)
    
    plt.figure(figsize=(12, 5))
    
    # Training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-o', label='Training Loss', linewidth=2)
    plt.plot(range(1, num_epochs + 1), val_losses, 'r-o', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Loss statistics
    plt.subplot(1, 2, 2)
    metrics_names = ['Final Train\nLoss', 'Final Val\nLoss', 'Min Train\nLoss', 'Min Val\nLoss']
    metrics_values = [
        train_losses[-1] if train_losses else 0,
        val_losses[-1] if val_losses else 0,
        min(train_losses) if train_losses else 0,
        min(val_losses) if val_losses else 0
    ]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Training Summary', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=9)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    training_plots_path = os.path.join(results_dir, 'training_summary.png')
    plt.savefig(training_plots_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training plots saved to: {training_plots_path}")
    plt.show()
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📊 Final Results:")
    print(f"  • Final Training Loss: {train_losses[-1]:.4f}")
    print(f"  • Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"  • Best Training Loss: {min(train_losses):.4f} (Epoch {train_losses.index(min(train_losses))+1})")
    print(f"  • Best Validation Loss: {min(val_losses):.4f} (Epoch {val_losses.index(min(val_losses))+1})")
    print(f"\n📁 Outputs saved to:")
    print(f"  • Model: {model_save_path}")
    print(f"  • Training plots: {training_plots_path}")
    print(f"\n🎯 Next steps:")
    print(f"  1. Review training plots in {results_dir}")
    print(f"  2. Run evaluation on test set")
    print(f"  3. Fine-tune hyperparameters if needed")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_path': model_save_path
    }


if __name__ == "__main__":
    results = main()

#main.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for SimSurgSkill dataset model training with COCO format data
"""
import os
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from src.models import EfficientDetModel
from src.coco_data_loader import get_coco_data_loaders
from src.evaluation_metrics import run_evaluation
import matplotlib.pyplot as plt

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
    
    # Check if COCO format data exists
    if not os.path.exists(coco_dir):
        print("\n❌ ERROR: COCO format data not found!")
        print(f"   Expected location: {coco_dir}")
        print("\n✅ SOLUTION: Run the data pipeline first:")
        print("   python data.py --data_dir data/simsurgskill_2021_dataset")
        return
    
    # Use existing COCO data (created by data.py)
    coco_paths = {
        'train_dir': os.path.join(coco_dir, 'train', 'images'),
        'val_dir': os.path.join(coco_dir, 'val', 'images'),
        'test_dir': os.path.join(coco_dir, 'test', 'images'),
        'train_ann': os.path.join(coco_dir, 'annotations', 'instances_train.json'),
        'val_ann': os.path.join(coco_dir, 'annotations', 'instances_val.json'),
        'test_ann': os.path.join(coco_dir, 'annotations', 'instances_test.json')
    }
    
    # Verify all paths exist
    print("\nVerifying COCO format data...")
    all_exist = True
    for key, path in coco_paths.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"  {status} {key}: {path}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print("\n❌ ERROR: Some COCO files are missing!")
        print("✅ SOLUTION: Run the data pipeline:")
        print("   python data.py --data_dir data/simsurgskill_2021_dataset")
        return
    
    print("\n✅ All COCO files found!")
    
    # Get data loaders for COCO format data
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = get_coco_data_loaders(
        coco_paths, 
        batch_size=8,
        num_workers=2,  # Reduce for Colab
        target_size=(720, 1280)  # Resize all images to same size
    )
    
    print(f"✅ Created data loaders:")
    print(f"   - Training batches: {len(train_loader)}")
    print(f"   - Validation batches: {len(val_loader)}")
    print(f"   - Test batches: {len(test_loader)}")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    
    model = EfficientDetModel(num_classes=2).to(device)  # 2 classes: needle, needle_driver
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("✅ Model initialized and ready for training")
    print(f"   - Classes: 2 (needle, needle_driver)")
    print(f"   - Optimizer: Adam (lr=0.001)")
    
    # Training loop
    num_epochs = 10
    train_losses = []
    val_losses = []
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            
            # Move targets to device
            for j in range(len(targets)):
                for k in targets[j]:
                    targets[j][k] = targets[j][k].to(device)
            
            # Forward pass
            outputs = model(images, targets)
            loss = outputs['loss']
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                
                # Move targets to device
                for j in range(len(targets)):
                    for k in targets[j]:
                        targets[j][k] = targets[j][k].to(device)
                
                # Forward pass
                outputs = model(images, targets)
                loss = outputs['loss']
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'\n📊 Epoch [{epoch+1}/{num_epochs}]:')
        print(f'   Training Loss:   {avg_train_loss:.4f}')
        print(f'   Validation Loss: {avg_val_loss:.4f}\n')
    
    # Save the model
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'✅ Model saved to {model_save_path}')
    
    # EVALUATION PHASE
    print("\n" + "="*60)
    print("STARTING EVALUATION PHASE")
    print("="*60)
    
    # Load the best model
    model.load_state_dict(torch.load(model_save_path))
    
    # Run comprehensive evaluation
    evaluation_results = run_evaluation(
        model=model,
        test_loader=test_loader,
        device=device,
        save_dir=results_dir
    )
    
    # Plot training curves
    print("\nGenerating training plots...")
    
    plt.figure(figsize=(12, 5))
    
    # Training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-o', label='Training Loss', linewidth=2)
    plt.plot(range(1, num_epochs + 1), val_losses, 'r-o', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Final metrics summary
    plt.subplot(1, 2, 2)
    metrics_names = ['Precision', 'Recall', 'mIoU']
    best_precision = max(evaluation_results['precision']) if evaluation_results['precision'] else 0
    best_recall = max(evaluation_results['recall']) if evaluation_results['recall'] else 0
    best_miou = max(evaluation_results['miou']) if evaluation_results['miou'] else 0
    metrics_values = [best_precision, best_recall, best_miou]
    
    bars = plt.bar(metrics_names, metrics_values, color=['blue', 'red', 'green'], alpha=0.7)
    plt.ylabel('Score')
    plt.title('Best Evaluation Metrics')
    plt.ylim([0, 1])
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    training_plots_path = os.path.join(results_dir, 'training_summary.png')
    plt.savefig(training_plots_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training plots saved to {training_plots_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📁 Results saved to: {results_dir}")
    print(f"📁 Model saved to: {model_save_path}")
    print(f"\n📊 Final Metrics:")
    print(f"   - Best Precision: {best_precision:.3f}")
    print(f"   - Best Recall:    {best_recall:.3f}")
    print(f"   - Best mIoU:      {best_miou:.3f}")
    
    return evaluation_results

if __name__ == "__main__":
    results = main()

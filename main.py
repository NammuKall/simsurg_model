#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for SimSurgSkill dataset processing and model training
with COCO format data
"""
import os
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from src.data_loader import process_videos_to_images
from src.visualization import visualize_metrics
from src.models import EfficientDetModel
from src.coco_data_loader import convert_to_coco_format, get_coco_data_loaders

def main():
    # Define paths - update these with your actual paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data/simsurgskill_2021_dataset")
    coco_dir = os.path.join(base_dir, "data/coco_format")
    
    # Process videos to images if needed
    process_v1 = True
    process_v2 = True
    process_test = True
    
    if process_v1:
        v1_dir = os.path.join(data_dir, "train_v1/videos/fps1/")
        process_videos_to_images(v1_dir)
    
    if process_v2:
        v2_dir = os.path.join(data_dir, "train_v2/videos/fps1/")
        process_videos_to_images(v2_dir)
    
    if process_test:
        test_dir = os.path.join(data_dir, "test/videos/fps1/")
        process_videos_to_images(test_dir)
    
    # Convert to COCO format
    convert_coco = True
    if convert_coco:
        coco_paths = convert_to_coco_format(data_dir, coco_dir)
        print(f"Dataset converted to COCO format at {coco_dir}")
    else:
        # If not converting, assume COCO data already exists
        coco_paths = {
            'train_dir': os.path.join(coco_dir, 'train', 'images'),
            'val_dir': os.path.join(coco_dir, 'val', 'images'),
            'test_dir': os.path.join(coco_dir, 'test', 'images'),
            'train_ann': os.path.join(coco_dir, 'annotations', 'instances_train.json'),
            'val_ann': os.path.join(coco_dir, 'annotations', 'instances_val.json'),
            'test_ann': os.path.join(coco_dir, 'annotations', 'instances_test.json')
        }
    
    # Load skill metrics data
    metrics_path = os.path.join(data_dir, "train_v1/annotations/skill_metric_gt.csv")
    skill_metric = pd.read_csv(metrics_path)
    print(f"Loaded {len(skill_metric)} skill metric records")
    
    # Visualize metrics
    visualize_metrics(skill_metric, 'needle_drop_counts', 'instrument_out_of_view_counts')
    
    # Get data loaders for COCO format data
    train_loader, val_loader, test_loader = get_coco_data_loaders(coco_paths, batch_size=8)
    print(f"Created data loaders with {len(train_loader)} training batches, {len(val_loader)} validation batches, and {len(test_loader)} test batches")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientDetModel(num_classes=3).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion_box = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Model initialized and ready for training")
    
    # Training loop
    num_epochs = 10
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')
        
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
    
    # Save the model
    model_save_path = os.path.join(base_dir, 'models', 'model.pth')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')
    
if __name__ == "__main__":
    main()

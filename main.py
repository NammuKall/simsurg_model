#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for SimSurgSkill dataset processing and model training
"""

import os
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from src.data_loader import load_train_data, process_videos_to_images
from src.visualization import visualize_metrics, visualize_bounding_box
from src.models import EfficientDetModel
from src.utils import IOU

def main():
    # Define paths - update these with your actual paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data/simsurgskill_2021_dataset")
    
    # Process videos to images if needed
    process_v1 = False
    process_v2 = False
    process_test = False
    
    if process_v1:
        v1_dir = os.path.join(data_dir, "train_v1/videos/fps1/")
        process_videos_to_images(v1_dir)
    
    if process_v2:
        v2_dir = os.path.join(data_dir, "train_v2/videos/fps1/")
        process_videos_to_images(v2_dir)
    
    if process_test:
        test_dir = os.path.join(data_dir, "test/videos/fps1/")
        process_videos_to_images(test_dir)
    
    # Load skill metrics data
    metrics_path = os.path.join(data_dir, "train_v1/annotations/skill_metric_gt.csv")
    skill_metric = pd.read_csv(metrics_path)
    print(f"Loaded {len(skill_metric)} skill metric records")
    
    # Visualize metrics
    visualize_metrics(skill_metric, 'needle_drop_counts', 'instrument_out_of_view_counts')
    
    # Load training data
    v1_data_dir = os.path.join(data_dir, "train_v1/videos/fps1/")
    v1_label_dir = os.path.join(data_dir, "train_v1/annotations/bounding_box_gt/")
    v1_train_data, v1_train_array = load_train_data(v1_data_dir)
    print(f"v1 training data shape: {v1_train_array.shape}")
    
    v2_data_dir = os.path.join(data_dir, "train_v2/videos/fps1/")
    v2_train_data, v2_train_array = load_train_data(v2_data_dir, resize=True)
    print(f"v2 training data shape: {v2_train_array.shape}")
    
    test_data_dir = os.path.join(data_dir, "test/videos/fps1/")
    test_train_data, test_train_array = load_train_data(test_data_dir)
    print(f"test data shape: {test_train_array.shape}")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientDetModel(num_classes=3).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion_box = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Model initialized and ready for training")
    
    # Note: Training loop would be implemented here
    
if __name__ == "__main__":
    main()

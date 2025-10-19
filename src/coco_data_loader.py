#coco_data_loader.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO format data loader for SimSurgSkill dataset - FIXED VERSION
"""
import os
import cv2
import json
import numpy as np
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class COCODataset(Dataset):
    """PyTorch Dataset for COCO format data"""
    def __init__(self, root_dir, annotation_file, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images
            annotation_file (string): Path to COCO annotation file
            transform (callable, optional): Transform to apply to samples
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
            
        # Create image id to annotations mapping
        self.image_ids = []
        self.annotations_by_image = {}
        
        for img in self.coco_data['images']:
            image_id = img['id']
            self.image_ids.append(image_id)
            self.annotations_by_image[image_id] = []
        
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id in self.annotations_by_image:
                self.annotations_by_image[image_id].append(ann)
                
        # Create category id to name mapping
        self.categories = {}
        for cat in self.coco_data['categories']:
            self.categories[cat['id']] = cat['name']
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Find image info
        image_info = None
        for img in self.coco_data['images']:
            if img['id'] == image_id:
                image_info = img
                break
                
        # Load image
        img_name = image_info['file_name']
        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Get annotations for this image
        annotations = self.annotations_by_image[image_id]
        
        # Extract bounding boxes, labels, etc.
        boxes = []
        labels = []
        
        for ann in annotations:
            # COCO format is [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Convert to [x_min, y_min, x_max, y_max] format
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros(0, dtype=np.int64)
        
        # Create sample dict
        sample = {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class Compose:
    """Composes several transforms together"""
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor:
    """Convert ndarrays in sample to Tensors with resizing"""
    def __init__(self, target_size=(720, 1280)):
        """
        Args:
            target_size: (height, width) to resize images to
        """
        self.target_size = target_size
    
    def __call__(self, sample):
        image, boxes, labels = sample['image'], sample['boxes'], sample['labels']
        
        # Get original dimensions
        orig_height, orig_width = image.shape[:2]
        target_height, target_width = self.target_size
        
        # Resize image
        image = cv2.resize(image, (target_width, target_height))
        
        # Scale bounding boxes
        if len(boxes) > 0:
            scale_x = target_width / orig_width
            scale_y = target_height / orig_height
            
            # boxes format is [x_min, y_min, x_max, y_max]
            boxes[:, 0] *= scale_x  # x_min
            boxes[:, 1] *= scale_y  # y_min
            boxes[:, 2] *= scale_x  # x_max
            boxes[:, 3] *= scale_y  # y_max
        
        # Convert image to tensor
        image = image.transpose((2, 0, 1))  # Convert to (C, H, W)
        sample['image'] = torch.from_numpy(image).float() / 255.0  # Normalize
        
        # Convert boxes and labels to tensor
        sample['boxes'] = torch.from_numpy(boxes)
        sample['labels'] = torch.from_numpy(labels)
        sample['image_id'] = torch.tensor([sample['image_id']])
            
        return sample


def collate_fn(batch):
    """
    Collate function for DataLoader to handle samples with varying number of boxes
    """
    images = []
    targets = []
    
    for sample in batch:
        images.append(sample['image'])
        target = {
            'boxes': sample['boxes'],
            'labels': sample['labels'],
            'image_id': sample['image_id']
        }
        targets.append(target)
    
    images = torch.stack(images, 0)
    
    return images, targets


def get_coco_data_loaders(coco_paths, batch_size=8, num_workers=4, target_size=(720, 1280)):
    """
    Get PyTorch DataLoaders for COCO dataset
    
    Args:
        coco_paths (dict): Dictionary with paths to COCO directories and files
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of workers for DataLoader
        target_size (tuple): (height, width) to resize all images to
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Define transforms with resizing
    transform = Compose([
        ToTensor(target_size=target_size)
    ])
    
    # Create datasets
    train_dataset = COCODataset(
        coco_paths['train_dir'],
        coco_paths['train_ann'],
        transform=transform
    )
    
    val_dataset = COCODataset(
        coco_paths['val_dir'],
        coco_paths['val_ann'],
        transform=transform
    )
    
    test_dataset = COCODataset(
        coco_paths['test_dir'],
        coco_paths['test_ann'],
        transform=transform
    )
    
    print(f"Created datasets with {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test samples")
    print(f"Images will be resized to: {target_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader

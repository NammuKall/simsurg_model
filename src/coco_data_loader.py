#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO format data loader for SimSurgSkill dataset
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

def convert_to_coco_format(data_dir, output_dir, train_val_split=0.8):
    """
    Convert existing dataset to COCO format
    
    Args:
        data_dir (str): Directory containing SimSurgSkill dataset
        output_dir (str): Output directory for COCO format data
        train_val_split (float): Ratio of training to validation data
    
    Returns:
        dict: Dictionary with paths to COCO directories and files
    """
    # Create COCO directory structure
    os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
    
    # Define categories
    categories = [
        {"id": 1, "name": "instrument", "supercategory": "instrument"},
        {"id": 2, "name": "needle", "supercategory": "needle"},
        {"id": 3, "name": "thread", "supercategory": "thread"}
    ]
    
    # Process data directories
    print("Processing dataset directories...")
    v1_images, v1_annotations = process_dataset_dir(
        os.path.join(data_dir, "train_v1/videos/fps1/"),
        os.path.join(data_dir, "train_v1/annotations/bounding_box_gt/")
    )
    
    v2_images, v2_annotations = process_dataset_dir(
        os.path.join(data_dir, "train_v2/videos/fps1/"),
        os.path.join(data_dir, "train_v2/annotations/bounding_box_gt/")
    )
    
    test_images, test_annotations = process_dataset_dir(
        os.path.join(data_dir, "test/videos/fps1/"),
        os.path.join(data_dir, "test/annotations/bounding_box_gt/")
    )
    
    print(f"Found {len(v1_images)} images in train_v1, {len(v2_images)} in train_v2, {len(test_images)} in test")
    
    # Combine train_v1 and train_v2 for training data
    all_train_images = v1_images + v2_images
    all_train_annotations = {**v1_annotations, **v2_annotations}
    
    # Split training data into train and validation sets
    np.random.seed(42)
    indices = np.arange(len(all_train_images))
    np.random.shuffle(indices)
    split_idx = int(len(all_train_images) * train_val_split)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_images = [all_train_images[i] for i in train_indices]
    val_images = [all_train_images[i] for i in val_indices]
    
    print(f"Split data into {len(train_images)} training and {len(val_images)} validation images")
    
    # Create COCO datasets
    print("Creating COCO format datasets...")
    create_coco_dataset(
        train_images, 
        all_train_annotations, 
        categories, 
        os.path.join(output_dir, 'train', 'images'),
        os.path.join(output_dir, 'annotations', 'instances_train.json')
    )
    
    create_coco_dataset(
        val_images, 
        all_train_annotations, 
        categories, 
        os.path.join(output_dir, 'val', 'images'),
        os.path.join(output_dir, 'annotations', 'instances_val.json')
    )
    
    create_coco_dataset(
        test_images, 
        test_annotations, 
        categories, 
        os.path.join(output_dir, 'test', 'images'),
        os.path.join(output_dir, 'annotations', 'instances_test.json')
    )
    
    # Return paths dictionary
    return {
        'train_dir': os.path.join(output_dir, 'train', 'images'),
        'val_dir': os.path.join(output_dir, 'val', 'images'),
        'test_dir': os.path.join(output_dir, 'test', 'images'),
        'train_ann': os.path.join(output_dir, 'annotations', 'instances_train.json'),
        'val_ann': os.path.join(output_dir, 'annotations', 'instances_val.json'),
        'test_ann': os.path.join(output_dir, 'annotations', 'instances_test.json')
    }

def process_dataset_dir(image_dir, annotation_dir):
    """
    Process a dataset directory to extract images and annotations
    
    Args:
        image_dir (str): Directory containing images
        annotation_dir (str): Directory containing annotations
    
    Returns:
        tuple: (list of image paths, dictionary of annotations)
    """
    images = []
    annotations = {}
    
    # Check if directories exist
    if not os.path.exists(image_dir):
        print(f"Warning: Image directory {image_dir} does not exist")
        return images, annotations
        
    if not os.path.exists(annotation_dir):
        print(f"Warning: Annotation directory {annotation_dir} does not exist")
        return images, annotations
    
    # Get all image files
    for file_name in os.listdir(image_dir):
        if file_name.endswith('.jpeg'):
            img_path = os.path.join(image_dir, file_name)
            images.append(img_path)
            
            # Find corresponding annotation file
            base_name = os.path.splitext(file_name)[0]
            ann_file = os.path.join(annotation_dir, f"{base_name}.json")
            
            if os.path.exists(ann_file):
                try:
                    with open(ann_file, 'r') as f:
                        annotations[img_path] = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse annotation file {ann_file}")
    
    return images, annotations

def create_coco_dataset(image_paths, annotations, categories, output_img_dir, output_ann_file):
    """
    Create a COCO format dataset from images and annotations
    
    Args:
        image_paths (list): List of image paths
        annotations (dict): Dictionary of annotations by image path
        categories (list): List of category dictionaries
        output_img_dir (str): Output directory for images
        output_ann_file (str): Output file for annotations
    
    Returns:
        dict: COCO format dataset
    """
    # Initialize COCO format structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    image_id = 0
    annotation_id = 0
    
    # Process each image
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        
        try:
            # Copy image to output directory
            shutil.copy(img_path, os.path.join(output_img_dir, img_name))
            
            # Read image to get dimensions
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}, skipping")
                continue
                
            height, width = img.shape[:2]
            
            # Add image info to COCO format
            coco_data["images"].append({
                "id": image_id,
                "file_name": img_name,
                "width": width,
                "height": height
            })
            
            # Process annotations for this image
            if img_path in annotations:
                for obj_id, ann in annotations[img_path].items():
                    if "x_min" in ann and "y_min" in ann and "x_max" in ann and "y_max" in ann:
                        # Extract bounding box coordinates
                        x_min = float(ann["x_min"])
                        y_min = float(ann["y_min"])
                        x_max = float(ann["x_max"])
                        y_max = float(ann["y_max"])
                        
                        # COCO format uses [x, y, width, height]
                        width = x_max - x_min
                        height = y_max - y_min
                        
                        # Determine category_id
                        category_id = 1  # Default
                        if "class" in ann:
                            class_name = ann["class"]
                            for cat in categories:
                                if cat["name"] == class_name:
                                    category_id = cat["id"]
                                    break
                        
                        # Create COCO annotation
                        coco_annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [x_min, y_min, width, height],
                            "area": width * height,
                            "iscrowd": 0
                        }
                        
                        coco_data["annotations"].append(coco_annotation)
                        annotation_id += 1
            
            image_id += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Write annotations to file
    with open(output_ann_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Created COCO dataset with {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
    return coco_data

class Compose:
    """Composes several transforms together"""
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

class ToTensor:
    """Convert ndarrays in sample to Tensors"""
    def __call__(self, sample):
        image, boxes, labels = sample['image'], sample['boxes'], sample['labels']
        
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

def get_coco_data_loaders(coco_paths, batch_size=8, num_workers=4):
    """
    Get PyTorch DataLoaders for COCO dataset
    
    Args:
        coco_paths (dict): Dictionary with paths to COCO directories and files
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of workers for DataLoader
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Define transforms
    transform = Compose([
        ToTensor()
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

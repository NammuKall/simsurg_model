#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert SimSurgSkill dataset to COCO format
"""
import os
import json
import shutil
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_coco_directory_structure(base_dir):
    """
    Create COCO format directory structure
    
    Args:
        base_dir (str): Base directory where the COCO structure will be created
    """
    coco_dir = os.path.join(base_dir, "coco_format")
    
    # Create main directories
    os.makedirs(coco_dir, exist_ok=True)
    os.makedirs(os.path.join(coco_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(coco_dir, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(coco_dir, "test", "images"), exist_ok=True)
    os.makedirs(os.path.join(coco_dir, "annotations"), exist_ok=True)
    
    print(f"Created COCO directory structure at {coco_dir}")
    return coco_dir

def convert_bbox_to_coco(annotation, image_id, annotation_id, category_id=1):
    """
    Convert bounding box annotation to COCO format
    
    Args:
        annotation (dict): Original annotation
        image_id (int): Image ID
        annotation_id (int): Annotation ID
        category_id (int): Category ID
        
    Returns:
        dict: COCO format annotation
    """
    # Extract bounding box coordinates
    x_min = annotation.get("x_min", 0)
    y_min = annotation.get("y_min", 0)
    x_max = annotation.get("x_max", 0)
    y_max = annotation.get("y_max", 0)
    
    # COCO format uses [x,y,width,height]
    width = x_max - x_min
    height = y_max - y_min
    
    # Create COCO annotation
    coco_annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [float(x_min), float(y_min), float(width), float(height)],
        "area": float(width * height),
        "iscrowd": 0
    }
    
    return coco_annotation

def create_coco_annotations(annotations_dir, image_dir, output_file, categories):
    """
    Create COCO format annotations JSON file
    
    Args:
        annotations_dir (str): Directory containing original annotations
        image_dir (str): Directory containing images
        output_file (str): Path to output JSON file
        categories (list): List of category dictionaries
        
    Returns:
        dict: COCO format annotations
    """
    # Initialize COCO format structure
    coco_json = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    image_id = 0
    annotation_id = 0
    
    # Get all annotation files
    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
    
    for ann_file in annotation_files:
        ann_path = os.path.join(annotations_dir, ann_file)
        
        # Load annotation file
        with open(ann_path, 'r') as f:
            original_annotations = json.load(f)
        
        # Get corresponding image filename (assuming same base name)
        image_base_name = os.path.splitext(ann_file)[0]
        image_file = f"{image_base_name}.jpeg"
        image_path = os.path.join(image_dir, image_file)
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping annotation {ann_file}")
            continue
        
        # Get image dimensions
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        # Add image info to COCO format
        coco_json["images"].append({
            "id": image_id,
            "file_name": image_file,
            "width": width,
            "height": height
        })
        
        # Convert each annotation to COCO format
        for obj_id, ann in original_annotations.items():
            if "x_min" in ann and "y_min" in ann and "x_max" in ann and "y_max" in ann:
                # Determine category_id based on annotation class if available
                category_id = 1  # Default to 1 if class info not available
                if "class" in ann:
                    class_name = ann["class"]
                    # Find corresponding category_id
                    for cat in categories:
                        if cat["name"] == class_name:
                            category_id = cat["id"]
                            break
                
                coco_ann = convert_bbox_to_coco(ann, image_id, annotation_id, category_id)
                coco_json["annotations"].append(coco_ann)
                annotation_id += 1
        
        image_id += 1
    
    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(coco_json, f, indent=2)
    
    print(f"Created COCO annotations at {output_file}")
    return coco_json

def split_and_organize_data(base_dir, data_dir, coco_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data into train/val/test sets and organize into COCO format
    
    Args:
        base_dir (str): Base directory
        data_dir (str): Directory containing SimSurgSkill dataset
        coco_dir (str): COCO format output directory
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
    """
    # Define categories for COCO format
    # Update these based on your actual classes
    categories = [
        {"id": 1, "name": "instrument", "supercategory": "instrument"},
        {"id": 2, "name": "needle", "supercategory": "needle"},
        {"id": 3, "name": "thread", "supercategory": "thread"}
    ]
    
    # Gather all image files from train_v1, train_v2 and test directories
    all_images = []
    
    v1_image_dir = os.path.join(data_dir, "train_v1/videos/fps1/")
    v1_label_dir = os.path.join(data_dir, "train_v1/annotations/bounding_box_gt/")
    for f in os.listdir(v1_image_dir):
        if f.endswith('.jpeg'):
            all_images.append({
                "image_path": os.path.join(v1_image_dir, f),
                "annotation_dir": v1_label_dir,
                "set": "train_v1"
            })
    
    v2_image_dir = os.path.join(data_dir, "train_v2/videos/fps1/")
    v2_label_dir = os.path.join(data_dir, "train_v2/annotations/bounding_box_gt/")
    for f in os.listdir(v2_image_dir):
        if f.endswith('.jpeg'):
            all_images.append({
                "image_path": os.path.join(v2_image_dir, f),
                "annotation_dir": v2_label_dir,
                "set": "train_v2"
            })
    
    test_image_dir = os.path.join(data_dir, "test/videos/fps1/")
    test_label_dir = os.path.join(data_dir, "test/annotations/bounding_box_gt/")
    for f in os.listdir(test_image_dir):
        if f.endswith('.jpeg'):
            all_images.append({
                "image_path": os.path.join(test_image_dir, f),
                "annotation_dir": test_label_dir,
                "set": "test"
            })
    
    # Split data
    if test_ratio > 0:
        # Use existing test set if it exists
        train_val = [img for img in all_images if img["set"] != "test"]
        test_set = [img for img in all_images if img["set"] == "test"]
        
        # If test set is empty or too small, use a random split
        if len(test_set) < len(all_images) * test_ratio * 0.5:
            train_val, test_set = train_test_split(all_images, test_size=test_ratio, random_state=42)
    else:
        train_val = all_images
        test_set = []
    
    # Split train_val into train and val
    if val_ratio > 0:
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_set, val_set = train_test_split(train_val, test_size=val_ratio_adjusted, random_state=42)
    else:
        train_set = train_val
        val_set = []
    
    # Copy images and organize by set
    train_dir = os.path.join(coco_dir, "train", "images")
    val_dir = os.path.join(coco_dir, "val", "images")
    test_dir = os.path.join(coco_dir, "test", "images")
    
    # Keep track of images in each set for annotation creation
    train_images = []
    val_images = []
    test_images = []
    
    # Process training set
    for img in train_set:
        img_path = img["image_path"]
        img_name = os.path.basename(img_path)
        dst_path = os.path.join(train_dir, img_name)
        shutil.copy(img_path, dst_path)
        train_images.append({
            "path": dst_path,
            "name": img_name,
            "annotation_dir": img["annotation_dir"]
        })
    
    # Process validation set
    for img in val_set:
        img_path = img["image_path"]
        img_name = os.path.basename(img_path)
        dst_path = os.path.join(val_dir, img_name)
        shutil.copy(img_path, dst_path)
        val_images.append({
            "path": dst_path,
            "name": img_name,
            "annotation_dir": img["annotation_dir"]
        })
    
    # Process test set
    for img in test_set:
        img_path = img["image_path"]
        img_name = os.path.basename(img_path)
        dst_path = os.path.join(test_dir, img_name)
        shutil.copy(img_path, dst_path)
        test_images.append({
            "path": dst_path,
            "name": img_name,
            "annotation_dir": img["annotation_dir"]
        })
    
    print(f"Organized {len(train_images)} training images, {len(val_images)} validation images, and {len(test_images)} test images")
    
    # Create COCO annotations for each set
    annotations_dir = os.path.join(coco_dir, "annotations")
    
    # Gather all annotation files for each set
    train_annotations = {}
    for img in train_images:
        ann_name = os.path.splitext(img["name"])[0] + ".json"
        ann_path = os.path.join(img["annotation_dir"], ann_name)
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                train_annotations[img["name"]] = json.load(f)
    
    val_annotations = {}
    for img in val_images:
        ann_name = os.path.splitext(img["name"])[0] + ".json"
        ann_path = os.path.join(img["annotation_dir"], ann_name)
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                val_annotations[img["name"]] = json.load(f)
    
    test_annotations = {}
    for img in test_images:
        ann_name = os.path.splitext(img["name"])[0] + ".json"
        ann_path = os.path.join(img["annotation_dir"], ann_name)
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                test_annotations[img["name"]] = json.load(f)
    
    # Create COCO format annotations
    create_coco_annotations_by_set(train_dir, train_annotations, os.path.join(annotations_dir, "instances_train.json"), categories)
    create_coco_annotations_by_set(val_dir, val_annotations, os.path.join(annotations_dir, "instances_val.json"), categories)
    create_coco_annotations_by_set(test_dir, test_annotations, os.path.join(annotations_dir, "instances_test.json"), categories)

def create_coco_annotations_by_set(image_dir, annotations_dict, output_file, categories):
    """
    Create COCO format annotations JSON file from a dictionary of annotations
    
    Args:
        image_dir (str): Directory containing images
        annotations_dict (dict): Dictionary of annotations by image name
        output_file (str): Path to output JSON file
        categories (list): List of category dictionaries
        
    Returns:
        dict: COCO format annotations
    """
    # Initialize COCO format structure
    coco_json = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    image_id = 0
    annotation_id = 0
    
    # Process each image with annotations
    for image_file, annotations in annotations_dict.items():
        image_path = os.path.join(image_dir, image_file)
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping annotation")
            continue
        
        # Get image dimensions
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}, skipping")
            continue
            
        height, width = img.shape[:2]
        
        # Add image info to COCO format
        coco_json["images"].append({
            "id": image_id,
            "file_name": image_file,
            "width": width,
            "height": height
        })
        
        # Convert each annotation to COCO format
        for obj_id, ann in annotations.items():
            if "x_min" in ann and "y_min" in ann and "x_max" in ann and "y_max" in ann:
                # Determine category_id based on annotation class if available
                category_id = 1  # Default to 1 if class info not available
                if "class" in ann:
                    class_name = ann["class"]
                    # Find corresponding category_id
                    for cat in categories:
                        if cat["name"] == class_name:
                            category_id = cat["id"]
                            break
                
                coco_ann = convert_bbox_to_coco(ann, image_id, annotation_id, category_id)
                coco_json["annotations"].append(coco_ann)
                annotation_id += 1
        
        image_id += 1
    
    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(coco_json, f, indent=2)
    
    print(f"Created COCO annotations at {output_file} with {len(coco_json['images'])} images and {len(coco_json['annotations'])} annotations")
    return coco_json

def convert_to_coco_format(base_dir, data_dir):
    """
    Main function to convert SimSurgSkill dataset to COCO format
    
    Args:
        base_dir (str): Base directory
        data_dir (str): Directory containing SimSurgSkill dataset
    """
    # Create COCO directory structure
    coco_dir = create_coco_directory_structure(base_dir)
    
    # Split and organize data
    split_and_organize_data(base_dir, data_dir, coco_dir)
    
    print(f"Conversion to COCO format completed. Output directory: {coco_dir}")
    return coco_dir

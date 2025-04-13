#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data loading and preprocessing functions for the SimSurgSkill dataset
"""

import os
import cv2
import json
import numpy as np

def process_videos_to_images(video_dir):
    """
    Convert videos in the specified directory to JPEG images
    
    Args:
        video_dir (str): Directory containing video files
    """
    print(f"Processing videos in {video_dir}...")
    for video_name in os.listdir(video_dir):
        if "mp4" in video_name:
            video_path = os.path.join(video_dir, video_name)
            
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            while frame_count < 1:
                # Read a frame from the video
                ret, frame = cap.read()
                
                if ret:
                    frame_count += 1
                    # Define the filename for the current frame
                    image_name = os.path.splitext(video_name)[0] + ".jpeg"
                    frame_filename = os.path.join(video_dir, image_name)
                    
                    # Save the frame as an image file
                    cv2.imwrite(frame_filename, frame)
                    print(f'Saved: {frame_filename}')
            
            # Release the video capture object
            cap.release()

def load_train_data(image_dir, resize=False):
    """
    Load training data from the specified directory
    
    Args:
        image_dir (str): Directory containing image files
        resize (bool): Whether to resize images to 720x1280
    
    Returns:
        tuple: (image_paths, image_array)
    """
    train_data = []
    for file_name in os.listdir(image_dir):
        if "jpeg" in file_name:
            complete_path = os.path.join(image_dir, file_name)
            train_data.append(complete_path)
    
    train_array = np.zeros((len(train_data), 720, 1280, 3))
    counter = 0
    for image in train_data:
        img = cv2.imread(image)
        if resize and img.shape != (720, 1280, 3):
            img = cv2.resize(img, (1280, 720))
        train_array[counter,:,:,:] = img
        counter += 1
    
    return train_data, train_array

def load_bounding_box_annotations(label_dir):
    """
    Load bounding box annotations from JSON files
    
    Args:
        label_dir (str): Directory containing annotation files
    
    Returns:
        list: List of annotation dictionaries
    """
    annotations = []
    
    for label_path in os.listdir(label_dir):
        ann_path = os.path.join(label_dir, label_path)
        with open(ann_path, "rb") as file:
            temp_data = json.load(file)
            annotations.append(temp_data)
    
    return annotations

def image_load(image_path, target_size=(720, 1280)):
    """
    Load and resize an image
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing
    
    Returns:
        numpy.ndarray: The loaded and resized image
    """
    image = cv2.imread(image_path)
    if image is not None and target_size is not None:
        image = cv2.resize(image, (target_size[1], target_size[0]))
    return image

def label_load(label_path, frame_id):
    """
    Load a specific label for a frame
    
    Args:
        label_path (str): Path to the label file
        frame_id (int): ID of the frame to get label for
    
    Returns:
        dict: Label information for the frame
    """
    with open(label_path, "r") as file:
        labels = json.load(file)
        frame_labels = []
        for label in labels.keys():
            if labels[label]["frame_id"] == frame_id:
                frame_labels.append(labels[label])
        return frame_labels

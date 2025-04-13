#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for the SimSurgSkill dataset
"""

def IOU(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1 (tuple): (x1, y1, x2, y2) coordinates of first box
        box2 (tuple): (x3, y3, x4, y4) coordinates of second box
    
    Returns:
        float: IoU value
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    # Calculate intersection coordinates
    x_inter1 = max(x1, x3)
    y_inter1 = max(y1, y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)
    
    # Calculate area of intersection
    width_inter = max(0, x_inter2 - x_inter1)
    height_inter = max(0, y_inter2 - y_inter1)
    area_inter = width_inter * height_inter
    
    # Calculate areas of each box
    width_box1 = abs(x2 - x1)
    height_box1 = abs(y2 - y1)
    width_box2 = abs(x4 - x3)
    height_box2 = abs(y4 - y3)
    
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    
    # Calculate union area
    area_union = area_box1 + area_box2 - area_inter
    
    # Calculate IoU
    if area_union == 0:
        return 0
    
    iou = area_inter / area_union
    return iou

def convert_bbox_format(bbox, from_format='xyxy', to_format='xywh'):
    """
    Convert between bounding box formats
    
    Args:
        bbox (tuple): Bounding box coordinates
        from_format (str): Input format ('xyxy' or 'xywh')
        to_format (str): Output format ('xyxy' or 'xywh')
    
    Returns:
        tuple: Converted bounding box coordinates
    """
    if from_format == to_format:
        return bbox
    
    if from_format == 'xyxy' and to_format == 'xywh':
        x1, y1, x2, y2 = bbox
        return (x1, y1, x2 - x1, y2 - y1)
    
    elif from_format == 'xywh' and to_format == 'xyxy':
        x, y, w, h = bbox
        return (x, y, x + w, y + h)
    
    else:
        raise ValueError(f"Unsupported format conversion from {from_format} to {to_format}")

#testing.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing and evaluation functions for model evaluation
"""
import numpy as np
import torch

from src.utils import IOU


def compute_detection_metrics(model, data_loader, device, num_samples=None):
    """
    Compute detection metrics (IoU, precision, recall, mAP) on validation set
    
    Args:
        model: The model to evaluate
        data_loader: Data loader for validation set
        device: Torch device
        num_samples: Number of samples to evaluate (None = all samples)
    
    Returns:
        dict: Dictionary of metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if num_samples and sample_count >= num_samples:
                break
                
            images = images.to(device)
            
            # Move targets to device for loss computation (but we won't use loss mode)
            val_targets = []
            for j in range(len(targets)):
                val_targets.append({
                    'boxes': targets[j]['boxes'].to(device),
                    'labels': targets[j]['labels'].to(device),
                    'image_id': targets[j]['image_id'].to(device) if 'image_id' in targets[j] else torch.tensor([j], device=device)
                })
            
            # Get predictions in inference mode (without targets)
            outputs = model(images)
            
            # Store predictions and targets
            for i in range(len(images)):
                if isinstance(outputs, list) and len(outputs) > i:
                    pred = outputs[i]
                    all_predictions.append({
                        'boxes': pred['boxes'].cpu(),
                        'scores': pred['scores'].cpu(),
                        'labels': pred['labels'].cpu()
                    })
                    
                    all_targets.append({
                        'boxes': val_targets[i]['boxes'].cpu(),
                        'labels': val_targets[i]['labels'].cpu()
                    })
                    sample_count += 1
    
    if len(all_predictions) == 0:
        return {
            'mean_iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
            'true_positives': 0, 'false_positives': 0, 'false_negatives': 0
        }
    
    # Compute metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    iou_scores = []
    
    for pred, target in zip(all_predictions, all_targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        pred_labels = pred['labels']
        
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        
        if len(gt_boxes) == 0:
            false_positives += len(pred_boxes)
            continue
        
        # Match predictions to ground truths
        matched_gts = set()
        matched_preds = set()
        
        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            best_iou = 0.0
            best_pred_idx = -1
            
            # Convert label to scalar for comparison
            gt_label_val = gt_label.item() if isinstance(gt_label, torch.Tensor) else gt_label
            
            for pred_idx, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                if pred_idx in matched_preds:
                    continue
                
                # Convert label to scalar for comparison
                pred_label_val = pred_label.item() if isinstance(pred_label, torch.Tensor) else pred_label
                
                if pred_label_val != gt_label_val:
                    continue
                
                # Calculate IoU - handle both tensor and numpy inputs
                if isinstance(pred_box, torch.Tensor):
                    pred_box_np = pred_box.cpu().numpy()
                    # Flatten if needed (handle [1, 4] -> [4])
                    if pred_box_np.ndim > 1:
                        pred_box_np = pred_box_np.flatten()
                else:
                    pred_box_np = np.array(pred_box).flatten()
                if isinstance(gt_box, torch.Tensor):
                    gt_box_np = gt_box.cpu().numpy()
                    # Flatten if needed
                    if gt_box_np.ndim > 1:
                        gt_box_np = gt_box_np.flatten()
                else:
                    gt_box_np = np.array(gt_box).flatten()
                
                # Ensure we have exactly 4 values
                if len(pred_box_np) != 4 or len(gt_box_np) != 4:
                    continue
                
                iou = IOU(pred_box_np, gt_box_np)
                
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx
            
            if best_iou >= 0.5 and best_pred_idx >= 0:  # IoU threshold
                true_positives += 1
                matched_gts.add(gt_idx)
                matched_preds.add(best_pred_idx)
                iou_scores.append(best_iou)
            else:
                false_negatives += 1
        
        # Unmatched predictions are false positives
        false_positives += len(pred_boxes) - len(matched_preds)
    
    # Calculate metrics
    mean_iou = np.mean(iou_scores) if len(iou_scores) > 0 else 0.0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'mean_iou': float(mean_iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives)
    }


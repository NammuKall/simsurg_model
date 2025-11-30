#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faster R-CNN model implementation for object detection
Uses torchvision's Faster R-CNN with ResNet50 backbone
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNNModel(nn.Module):
    """
    Faster R-CNN model wrapper for SimSurgSkill dataset
    Uses ResNet50-FPN backbone with pretrained weights
    """
    
    def __init__(self, num_classes=2, pretrained=True, trainable_backbone_layers=3):
        """
        Initialize Faster R-CNN model
        
        Args:
            num_classes: Number of object classes (excluding background)
                         Default: 2 (needle, needle_driver)
                         Note: Background class is added automatically by Faster R-CNN
            pretrained: Whether to use pretrained weights
            trainable_backbone_layers: Number of trainable backbone layers (0-5)
        """
        super(FasterRCNNModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained Faster R-CNN with ResNet50-FPN backbone
        if pretrained:
            weights = 'DEFAULT'  # Uses COCO pretrained weights
        else:
            weights = None
        
        # Create Faster R-CNN model
        self.model = fasterrcnn_resnet50_fpn(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers
        )
        
        # Replace the classifier head to match number of classes
        # Faster R-CNN expects num_classes including background
        # So if we have 2 classes (needle, needle_driver), we need 3 total (background + 2)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # num_classes parameter excludes background, so we add 1 for background
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, 
            num_classes + 1  # +1 for background class (Faster R-CNN requirement)
        )
        
        # Store device for later use
        self._device = None
    
    def forward(self, x, targets=None):
        """
        Forward pass
        
        Args:
            x: Input images [batch_size, 3, H, W] or list of tensors
            targets: List of target dicts with 'boxes' and 'labels' (optional)
                    Each dict should have:
                    - 'boxes': tensor of shape [N, 4] in (x1, y1, x2, y2) format
                    - 'labels': tensor of shape [N] with class labels (1-indexed)
        
        Returns:
            If targets provided (training): dict with 'loss' key
            If inference (no targets): list of dicts with 'boxes', 'scores', 'labels'
        """
        # Convert single tensor to list if needed (Faster R-CNN expects list)
        if isinstance(x, torch.Tensor):
            x = [img for img in x]
        
        # Prepare targets format for torchvision Faster R-CNN
        # Faster R-CNN expects targets as list of dicts with 'boxes' and 'labels'
        if targets is not None:
            # Ensure targets are in correct format
            formatted_targets = []
            for target in targets:
                formatted_target = {
                    'boxes': target['boxes'].float(),
                    'labels': target['labels'].long()
                }
                formatted_targets.append(formatted_target)
            
            # FasterRCNN only returns loss dict when in training mode
            # In eval mode, it returns predictions even with targets
            # For validation loss computation, we need to temporarily enable training mode
            was_training = self.model.training
            try:
                # Temporarily set to training mode to compute loss
                self.model.train()
                loss_dict = self.model(x, formatted_targets)
                
                # Verify that loss_dict is actually a dict (not a list)
                if isinstance(loss_dict, dict):
                    # Sum all losses into a single scalar
                    total_loss = sum(loss_dict.values())
                    return {'loss': total_loss}
                else:
                    # This shouldn't happen - if it does, it means FasterRCNN returned unexpected type
                    raise ValueError(f"Expected loss dict when targets provided, got {type(loss_dict)}")
            finally:
                # Restore original training state
                if not was_training:
                    self.model.eval()
        else:
            # Inference mode: returns predictions
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(x)
            
            # Convert predictions to match our format
            # Faster R-CNN returns labels as 0-indexed (0=background, 1=class1, etc.)
            # We need to convert to 1-indexed (1=class1, 2=class2, etc.)
            formatted_predictions = []
            for pred in predictions:
                # Filter out background predictions (label 0)
                keep = pred['labels'] > 0
                formatted_pred = {
                    'boxes': pred['boxes'][keep],
                    'scores': pred['scores'][keep],
                    'labels': pred['labels'][keep]  # Already 1-indexed after filtering
                }
                formatted_predictions.append(formatted_pred)
            
            return formatted_predictions
    
    def train(self, mode=True):
        """Set model to training mode"""
        super().train(mode)
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set model to evaluation mode"""
        super().eval()
        self.model.eval()
        return self
    
    def to(self, device):
        """Move model to device"""
        super().to(device)
        self.model = self.model.to(device)
        self._device = device
        return self


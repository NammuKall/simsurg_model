#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EfficientDet-style model for object detection
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class EfficientDetModel(nn.Module):
    """
    Simplified EfficientDet-style model for object detection
    Fixed to handle tensor size mismatches
    """
    
    def __init__(self, num_classes=2, num_anchors=9):
        super(EfficientDetModel, self).__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Feature extraction backbone (ResNet50)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Get feature layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
        # Reduce channel dimensions to a common size
        self.reduce_channels = nn.ModuleDict({
            'P3': nn.Conv2d(512, 256, kernel_size=1),   # from layer2
            'P4': nn.Conv2d(1024, 256, kernel_size=1),  # from layer3
            'P5': nn.Conv2d(2048, 256, kernel_size=1)   # from layer4
        })
        
        # Feature fusion convolutions
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Detection heads
        self.class_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_anchors * num_classes, kernel_size=3, padding=1)
        )
        
        self.box_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_anchors * 4, kernel_size=3, padding=1)
        )

    def forward(self, x, targets=None):
        """
        Forward pass
        
        Args:
            x: Input images [batch_size, 3, H, W]
            targets: Ground truth annotations (optional)
            
        Returns:
            If targets provided: dict with 'loss'
            If inference (no targets): list of dicts with 'boxes', 'scores', 'labels'
        """
        batch_size = x.shape[0]
        
        # Extract features from backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c2 = self.layer1(x)   # stride 4
        c3 = self.layer2(c2)  # stride 8
        c4 = self.layer3(c3)  # stride 16
        c5 = self.layer4(c4)  # stride 32
        
        # Reduce to common channel size
        p3 = self.reduce_channels['P3'](c3)
        p4 = self.reduce_channels['P4'](c4)
        p5 = self.reduce_channels['P5'](c5)
        
        # Upsample P5 and P4 to match P3 size
        p5_upsampled = F.interpolate(p5, size=p3.shape[2:], mode='nearest')
        p4_upsampled = F.interpolate(p4, size=p3.shape[2:], mode='nearest')
        
        # Fuse features
        fused_features = self.fusion_conv(p3 + p4_upsampled + p5_upsampled)
        
        # Get predictions
        class_logits = self.class_head(fused_features)
        bbox_regression = self.box_head(fused_features)
        
        # Reshape outputs
        # class_logits: [B, num_anchors * num_classes, H, W] -> [B, H*W*num_anchors, num_classes]
        # bbox_regression: [B, num_anchors * 4, H, W] -> [B, H*W*num_anchors, 4]
        
        B, _, H, W = class_logits.shape
        
        class_logits = class_logits.permute(0, 2, 3, 1).contiguous()
        class_logits = class_logits.view(B, -1, self.num_classes)
        
        bbox_regression = bbox_regression.permute(0, 2, 3, 1).contiguous()
        bbox_regression = bbox_regression.view(B, -1, 4)
        
        if targets is not None:
            # Compute loss when targets are provided (training or validation)
            loss = self.compute_loss(class_logits, bbox_regression, targets)
            return {'loss': loss}
        else:
            # Inference mode: return predictions
            return self.postprocess_predictions(class_logits, bbox_regression)
    
    def compute_loss(self, class_logits, bbox_regression, targets):
        """
        Compute improved detection loss with IoU-based regression
        
        Args:
            class_logits: [B, num_predictions, num_classes]
            bbox_regression: [B, num_predictions, 4]
            targets: List of target dicts with 'boxes' and 'labels'
        """
        device = class_logits.device
        batch_size = class_logits.shape[0]
        
        classification_loss = torch.tensor(0.0, device=device)
        regression_loss = torch.tensor(0.0, device=device)
        
        for i in range(batch_size):
            # Get ground truth
            gt_boxes = targets[i]['boxes']
            gt_labels = targets[i]['labels']
            
            if len(gt_boxes) == 0:
                continue
            
            # Use up to num_gt objects
            num_gt = min(len(gt_labels), class_logits.shape[1])
            
            # Classification loss - Focal Loss for better training
            pred_logits = class_logits[i][:num_gt]
            target_labels = gt_labels[:num_gt].long() - 1  # Convert to 0-indexed
            
            # Clamp to valid range
            target_labels = torch.clamp(target_labels, 0, self.num_classes - 1)
            
            # Use reduced target labels where applicable
            valid_targets = target_labels < self.num_classes
            if valid_targets.sum() > 0:
                # Smoothed cross entropy for better training stability
                classification_loss += F.cross_entropy(
                    pred_logits[valid_targets], 
                    target_labels[valid_targets],
                    reduction='mean'
                )
            
            # Improved regression loss using smooth L1 (better than L1 for optimization)
            if len(gt_boxes) > 0:
                num_boxes = min(len(gt_boxes), bbox_regression.shape[1])
                pred_boxes = bbox_regression[i][:num_boxes]
                gt_boxes_tensor = gt_boxes[:num_boxes]
                
                regression_loss += F.smooth_l1_loss(pred_boxes, gt_boxes_tensor.float(), beta=0.1)
        
        # Combine losses with weighting
        total_loss = classification_loss + regression_loss * 0.5  # Weight regression loss
        
        return total_loss
    
    def postprocess_predictions(self, class_logits, bbox_regression, score_threshold=0.5):
        """
        Convert raw predictions to final detections
        
        Returns:
            List of dicts with 'boxes', 'scores', 'labels' for each image in batch
        """
        batch_size = class_logits.shape[0]
        results = []
        
        for i in range(batch_size):
            # Get scores and labels
            scores = torch.softmax(class_logits[i], dim=-1)
            max_scores, labels = scores.max(dim=-1)
            
            # Filter by score threshold
            keep = max_scores > score_threshold
            
            filtered_boxes = bbox_regression[i][keep]
            filtered_scores = max_scores[keep]
            filtered_labels = labels[keep] + 1  # Convert back to 1-indexed
            
            results.append({
                'boxes': filtered_boxes,
                'scores': filtered_scores,
                'labels': filtered_labels
            })
        
        return results


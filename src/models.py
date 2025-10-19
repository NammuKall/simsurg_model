#models.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural network model definitions for the SimSurgSkill dataset
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Basic residual block for ResNet"""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """ResNet model"""
    
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


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
        resnet = models.resnet50(pretrained=True)
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
            targets: Ground truth annotations (optional, for training)
            
        Returns:
            If training (targets provided): dict with 'loss'
            If inference: dict with 'boxes', 'scores', 'labels'
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
        
        if self.training and targets is not None:
            # Training mode: compute loss
            loss = self.compute_loss(class_logits, bbox_regression, targets)
            return {'loss': loss}
        else:
            # Inference mode: return predictions
            return self.postprocess_predictions(class_logits, bbox_regression)
    
    def compute_loss(self, class_logits, bbox_regression, targets):
        """
        Compute detection loss (simplified version)
        
        Args:
            class_logits: [B, num_predictions, num_classes]
            bbox_regression: [B, num_predictions, 4]
            targets: List of target dicts with 'boxes' and 'labels'
        """
        # This is a simplified loss - you should use proper detection loss
        # For now, we'll use a basic focal loss for classification
        
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
            
            # Simplified: just compute cross entropy on first prediction
            # In practice, you'd match predictions to ground truth
            pred_scores = torch.softmax(class_logits[i][:len(gt_labels)], dim=-1)
            target_labels = gt_labels.long() - 1  # Convert to 0-indexed
            
            # Clamp to valid range
            target_labels = torch.clamp(target_labels, 0, self.num_classes - 1)
            
            # Classification loss
            classification_loss += F.cross_entropy(
                class_logits[i][:len(gt_labels)], 
                target_labels
            )
            
            # Regression loss (L1)
            if len(gt_boxes) > 0:
                pred_boxes = bbox_regression[i][:len(gt_boxes)]
                regression_loss += F.l1_loss(pred_boxes, gt_boxes.float())
        
        # Average over batch
        total_loss = (classification_loss + regression_loss) / batch_size
        
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


# Convenience function to create model
def create_efficientdet_model(num_classes=2, pretrained=True):
    """
    Create EfficientDet model
    
    Args:
        num_classes: Number of object classes (default: 2 for needle and needle_driver)
        pretrained: Use pretrained ResNet50 backbone
    
    Returns:
        EfficientDetModel instance
    """
    return EfficientDetModel(num_classes=num_classes)

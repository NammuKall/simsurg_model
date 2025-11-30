#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet model definitions
"""

import torch
import torch.nn as nn
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
        self.num_classes = num_classes
        
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
        
        # Loss function for training
        self.criterion = nn.CrossEntropyLoss()

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

    def forward(self, x, targets=None):
        """
        Forward pass
        
        Args:
            x: Input images [batch_size, 3, H, W]
            targets: List of target dicts with 'boxes' and 'labels' (optional)
                    Used for training to compute loss
        
        Returns:
            If targets provided (training): dict with 'loss' key
            If inference (no targets): list of dicts with 'boxes', 'scores', 'labels'
        """
        # Extract features and get classification logits
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
        logits = self.fc(x)
        
        if targets is not None:
            # Training mode: compute classification loss
            # Convert detection targets to classification targets
            # Use the most common label in each image as the target
            batch_size = logits.shape[0]
            device = logits.device
            
            target_labels = []
            for i in range(batch_size):
                if len(targets[i]['labels']) > 0:
                    # Get the most common label (or first label if multiple)
                    labels = targets[i]['labels']
                    # Convert to list if tensor
                    if isinstance(labels, torch.Tensor):
                        label_list = labels.cpu().tolist()
                    else:
                        label_list = list(labels)
                    
                    # Find most common label (mode)
                    label_val = max(set(label_list), key=label_list.count)
                    
                    # Convert to 0-indexed classification label
                    # Handle both 0-indexed and 1-indexed label formats
                    # If min label is 0, assume 0-indexed; otherwise assume 1-indexed
                    min_label = min(label_list)
                    if min_label == 0:
                        # Already 0-indexed
                        label_idx = max(0, min(label_val, self.num_classes - 1))
                    else:
                        # 1-indexed, convert to 0-indexed
                        label_idx = max(0, min(label_val - 1, self.num_classes - 1))
                    target_labels.append(label_idx)
                else:
                    # No labels in image, use class 0 as default
                    target_labels.append(0)
            
            target_tensor = torch.tensor(target_labels, dtype=torch.long, device=device)
            loss = self.criterion(logits, target_tensor)
            
            return {'loss': loss}
        else:
            # Inference mode: return predictions in detection format
            # ResNet is a classification model, so we return empty detections
            # with classification scores as a fallback
            batch_size = logits.shape[0]
            predictions = []
            
            probs = F.softmax(logits, dim=1)
            
            for i in range(batch_size):
                # Get predicted class and confidence
                pred_class = torch.argmax(probs[i]).item()
                confidence = probs[i][pred_class].item()
                
                # Return empty boxes since ResNet doesn't do detection
                # But include the classification prediction for compatibility
                predictions.append({
                    'boxes': torch.empty((0, 4), dtype=torch.float32, device=logits.device),
                    'scores': torch.empty((0,), dtype=torch.float32, device=logits.device),
                    'labels': torch.empty((0,), dtype=torch.int64, device=logits.device)
                })
            
            return predictions


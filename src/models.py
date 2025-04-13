#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neural network model definitions for the SimSurgSkill dataset
"""

import torch
import torch.nn as nn
from torchvision import models

class ResidualBlock(nn.Module):
    """
    Basic residual block for ResNet
    """
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
    """
    ResNet model
    """
    def __init__(self, block, layers, num_classes=3):
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

class BiFPNBlock(nn.Module):
    """
    Bidirectional Feature Pyramid Network block for EfficientDet
    """
    def __init__(self, num_channels, epsilon=1e-4):
        super(BiFPNBlock, self).__init__()
        
        self.epsilon = epsilon
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, P3, P4, P5):
        # Upscale P5 and P4 to match P3's size
        P5_up = nn.functional.interpolate(P5, scale_factor=2, mode="nearest")
        P4_up = nn.functional.interpolate(P4, scale_factor=2, mode="nearest")
        
        # Combine feature maps and apply convolution
        P3_out = self.conv(P3 + P4_up + P5_up)
        return P3_out

class EfficientDetModel(nn.Module):
    """
    EfficientDet model for object detection
    """
    def __init__(self, num_classes=3):
        super(EfficientDetModel, self).__init__()
        
        # Feature extraction backbone (ResNet50)
        self.backbone = models.resnet50(pretrained=True)
        # Remove the classification head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # BiFPN block
        self.bifpn = BiFPNBlock(num_channels=2048)
        
        # Projection layer to reduce channels
        self.projection = nn.Conv2d(2048, 256, kernel_size=1)
        
        # Classification head
        self.class_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=3, padding=1)
        )
        
        # Bounding box regression head
        self.box_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Apply BiFPN (using the same feature map for each level in this simplified version)
        refined_features = self.bifpn(features, features, features)
        
        # Project to smaller number of channels
        projected_features = self.projection(refined_features)
        
        # Get classification and bounding box predictions
        class_logits = self.class_head(projected_features)
        bbox_regression = self.box_head(projected_features)
        
        return class_logits, bbox_regression

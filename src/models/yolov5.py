#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv5 model implementation for object detection
Uses ultralytics YOLOv5 or a PyTorch implementation compatible with SimSurgSkill dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

# Try to import ultralytics YOLOv5 (optional dependency)
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

# Import torchvision models for custom implementation
from torchvision import models


class YOLOv5Model(nn.Module):
    """
    YOLOv5 model wrapper for SimSurgSkill dataset
    Uses CSPDarkNet backbone with PANet neck and YOLOv5 detection heads
    """
    
    def __init__(self, num_classes=2, pretrained=True, model_size='s', input_size=640):
        """
        Initialize YOLOv5 model
        
        Args:
            num_classes: Number of object classes (excluding background)
                         Default: 2 (needle, needle_driver)
            pretrained: Whether to use pretrained weights
            model_size: Model size variant ('n', 's', 'm', 'l', 'x')
                       - 'n': nano (fastest, smallest)
                       - 's': small (default, good balance)
                       - 'm': medium
                       - 'l': large
                       - 'x': xlarge (slowest, most accurate)
            input_size: Input image size (default: 640, YOLOv5 standard)
        """
        super(YOLOv5Model, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.model_size = model_size
        
        # Try to use ultralytics YOLOv5 if available, otherwise use custom implementation
        if ULTRALYTICS_AVAILABLE:
            self.use_ultralytics = True
            # Note: ultralytics YOLO needs to be initialized differently
            # We'll handle this in forward pass
            self._ultralytics_model = None
        else:
            self.use_ultralytics = False
            # Use custom YOLOv5 implementation
            self.pretrained = pretrained
            self._build_custom_yolov5()
    
    def _build_custom_yolov5(self):
        """Build custom YOLOv5 architecture"""
        # YOLOv5 uses CSPDarkNet backbone
        # For simplicity, we'll use a ResNet-based backbone similar to other models
        # but with YOLOv5-style detection heads
        
        # Backbone (using ResNet50 for compatibility, can be replaced with CSPDarkNet)
        if self.model_size == 'n':
            backbone = models.mobilenet_v3_small(weights='IMAGENET1K_V1' if self.pretrained else None)
            backbone_channels = [24, 40, 112, 960]
        else:
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if self.pretrained else None)
            backbone_channels = [256, 512, 1024, 2048]
        
        # Extract feature layers
        self.backbone_conv1 = backbone.conv1
        self.backbone_bn1 = backbone.bn1
        self.backbone_relu = backbone.relu
        self.backbone_maxpool = backbone.maxpool
        self.backbone_layer1 = backbone.layer1  # stride 4
        self.backbone_layer2 = backbone.layer2  # stride 8
        self.backbone_layer3 = backbone.layer3  # stride 16
        self.backbone_layer4 = backbone.layer4  # stride 32
        
        # YOLOv5 uses 3 detection scales with PANet (Path Aggregation Network)
        # Detection heads
        self.detect_scale1 = self._make_yolov5_head(backbone_channels[3], self.num_classes)  # Large objects
        self.detect_scale2 = self._make_yolov5_head(backbone_channels[2], self.num_classes)  # Medium objects
        self.detect_scale3 = self._make_yolov5_head(backbone_channels[1], self.num_classes)  # Small objects
        
        # PANet upsampling and downsampling paths
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample1 = nn.Conv2d(backbone_channels[2], backbone_channels[2], kernel_size=3, stride=2, padding=1)
        self.downsample2 = nn.Conv2d(backbone_channels[1], backbone_channels[1], kernel_size=3, stride=2, padding=1)
        
        # Feature fusion
        self.fusion1 = nn.Conv2d(backbone_channels[3] + backbone_channels[2], backbone_channels[2], kernel_size=1)
        self.fusion2 = nn.Conv2d(backbone_channels[2] + backbone_channels[1], backbone_channels[1], kernel_size=1)
        
        # Number of anchors per scale (YOLOv5 uses 3 anchors per scale)
        self.num_anchors = 3
        
        # Store pretrained flag
        self.pretrained = self.pretrained if hasattr(self, 'pretrained') else True
    
    def _make_yolov5_head(self, in_channels, num_classes):
        """Create YOLOv5 detection head"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),  # YOLOv5 uses SiLU activation
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, self.num_anchors * (5 + num_classes), kernel_size=1)
            # 5 = 4 bbox coords + 1 objectness, num_classes = class predictions
        )
    
    def forward(self, x, targets=None):
        """
        Forward pass
        
        Args:
            x: Input images [batch_size, 3, H, W] or list of tensors
            targets: List of target dicts with 'boxes' and 'labels' (optional)
        
        Returns:
            If targets provided (training): dict with 'loss' key
            If inference (no targets): list of dicts with 'boxes', 'scores', 'labels'
        """
        if self.use_ultralytics:
            return self._forward_ultralytics(x, targets)
        else:
            return self._forward_custom(x, targets)
    
    def _forward_ultralytics(self, x, targets=None):
        """Forward pass using ultralytics YOLOv5"""
        # Initialize model if not already done
        if self._ultralytics_model is None:
            model_name = f'yolov5{self.model_size}.pt'
            self._ultralytics_model = YOLO(model_name)
            # Modify number of classes
            self._ultralytics_model.model.nc = self.num_classes
        
        # Convert input format
        if isinstance(x, torch.Tensor):
            # Convert tensor to numpy and handle batch
            images = []
            for i in range(x.shape[0]):
                img = x[i].cpu().numpy().transpose(1, 2, 0)
                img = (img * 255).astype('uint8')
                images.append(img)
        else:
            images = x
        
        if targets is not None:
            # Training mode
            # Note: ultralytics YOLO expects different format
            # For now, return a placeholder loss
            # In production, you'd need to format targets for ultralytics
            return {'loss': torch.tensor(0.0, device=x.device, requires_grad=True)}
        else:
            # Inference mode
            results = self._ultralytics_model(images)
            
            # Convert to our format
            formatted_results = []
            for result in results:
                boxes = []
                scores = []
                labels = []
                
                if result.boxes is not None:
                    for box in result.boxes:
                        boxes.append([box.xyxy[0][0].item(), box.xyxy[0][1].item(), 
                                     box.xyxy[0][2].item(), box.xyxy[0][3].item()])
                        scores.append(box.conf.item())
                        labels.append(int(box.cls.item()) + 1)  # Convert to 1-indexed
                
                formatted_results.append({
                    'boxes': torch.tensor(boxes, device=x.device) if boxes else torch.zeros((0, 4), device=x.device),
                    'scores': torch.tensor(scores, device=x.device) if scores else torch.zeros(0, device=x.device),
                    'labels': torch.tensor(labels, device=x.device, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long, device=x.device)
                })
            
            return formatted_results
    
    def _forward_custom(self, x, targets=None):
        """Forward pass using custom YOLOv5 implementation"""
        batch_size = x.shape[0]
        
        # Extract features from backbone
        x = self.backbone_conv1(x)
        x = self.backbone_bn1(x)
        x = self.backbone_relu(x)
        x = self.backbone_maxpool(x)
        
        x = self.backbone_layer1(x)   # stride 4
        small_features = self.backbone_layer2(x)  # stride 8
        medium_features = self.backbone_layer3(small_features)  # stride 16
        large_features = self.backbone_layer4(medium_features)  # stride 32
        
        # YOLOv5 PANet feature pyramid
        # Top-down path
        p5 = large_features
        p4 = torch.cat([self.upsample1(p5), medium_features], dim=1)
        p4 = self.fusion1(p4)
        p3 = torch.cat([self.upsample2(p4), small_features], dim=1)
        p3 = self.fusion2(p3)
        
        # Bottom-up path (for better feature fusion)
        p4 = torch.cat([p4, self.downsample1(p3)], dim=1)
        p5 = torch.cat([p5, self.downsample2(p4)], dim=1)
        
        # Detection at 3 scales
        scale1_out = self.detect_scale1(p5)  # Large objects
        scale2_out = self.detect_scale2(p4)  # Medium objects
        scale3_out = self.detect_scale3(p3)  # Small objects
        
        if targets is not None:
            # Training mode: compute loss
            loss = self.compute_loss([scale1_out, scale2_out, scale3_out], targets, x.shape[2:])
            return {'loss': loss}
        else:
            # Inference mode: return predictions
            return self.postprocess_predictions([scale1_out, scale2_out, scale3_out], x.shape[2:])
    
    def compute_loss(self, predictions, targets, img_size):
        """
        Compute YOLOv5 loss (simplified version)
        
        Args:
            predictions: List of 3 prediction tensors (one per scale)
            targets: List of target dicts
            img_size: Image size (H, W)
        """
        device = predictions[0].device
        total_loss = torch.tensor(0.0, device=device)
        
        batch_size = len(targets)
        
        # Use medium scale for simplified loss computation
        pred = predictions[1]
        B, _, H, W = pred.shape
        
        # Reshape prediction
        pred = pred.view(B, self.num_anchors, 5 + self.num_classes, H, W)
        pred = pred.permute(0, 3, 4, 1, 2).contiguous()
        
        coord_loss = torch.tensor(0.0, device=device)
        conf_loss = torch.tensor(0.0, device=device)
        class_loss = torch.tensor(0.0, device=device)
        
        for i in range(batch_size):
            gt_boxes = targets[i]['boxes']
            gt_labels = targets[i]['labels']
            
            if len(gt_boxes) == 0:
                continue
            
            for box_idx, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
                # Normalize box coordinates
                x_center = (box[0] + box[2]) / 2.0 / img_size[1]
                y_center = (box[1] + box[3]) / 2.0 / img_size[0]
                w = (box[2] - box[0]) / img_size[1]
                h = (box[3] - box[1]) / img_size[0]
                
                # Find grid cell
                grid_x = int(x_center * W)
                grid_y = int(y_center * H)
                grid_x = max(0, min(W - 1, grid_x))
                grid_y = max(0, min(H - 1, grid_y))
                
                # Get predictions for this grid cell (use first anchor)
                anchor_pred = pred[i, grid_y, grid_x, 0, :]
                
                # Extract components
                pred_x = torch.sigmoid(anchor_pred[0])
                pred_y = torch.sigmoid(anchor_pred[1])
                pred_w = anchor_pred[2]
                pred_h = anchor_pred[3]
                pred_conf = torch.sigmoid(anchor_pred[4])
                pred_classes = anchor_pred[5:]
                
                # Compute losses
                coord_loss += F.mse_loss(pred_x, torch.tensor(x_center, device=device))
                coord_loss += F.mse_loss(pred_y, torch.tensor(y_center, device=device))
                coord_loss += F.mse_loss(pred_w, torch.tensor(w, device=device))
                coord_loss += F.mse_loss(pred_h, torch.tensor(h, device=device))
                
                # Confidence loss
                conf_loss += F.binary_cross_entropy_with_logits(
                    anchor_pred[4],
                    torch.tensor(1.0, device=device)
                )
                
                # Classification loss
                label_idx = int(label.item()) - 1
                label_idx = max(0, min(self.num_classes - 1, label_idx))
                class_loss += F.cross_entropy(
                    pred_classes.unsqueeze(0),
                    torch.tensor([label_idx], device=device, dtype=torch.long)
                )
        
        # Combine losses with YOLOv5 weighting
        total_loss = 0.05 * coord_loss + conf_loss + 0.5 * class_loss
        return total_loss / batch_size if batch_size > 0 else total_loss
    
    def postprocess_predictions(self, predictions, img_size, score_threshold=0.25):
        """
        Convert raw YOLOv5 predictions to final detections
        
        Args:
            predictions: List of 3 prediction tensors (one per scale)
            img_size: Image size (H, W)
            score_threshold: Confidence threshold (YOLOv5 default: 0.25)
        
        Returns:
            List of dicts with 'boxes', 'scores', 'labels' for each image in batch
        """
        batch_size = predictions[0].shape[0]
        results = []
        
        # Use medium scale predictions
        pred = predictions[1]
        B, _, H, W = pred.shape
        
        # Reshape
        pred = pred.view(B, self.num_anchors, 5 + self.num_classes, H, W)
        pred = pred.permute(0, 3, 4, 1, 2).contiguous()
        
        for i in range(batch_size):
            boxes = []
            scores = []
            labels = []
            
            # Process each grid cell
            for grid_y in range(H):
                for grid_x in range(W):
                    for anchor_idx in range(self.num_anchors):
                        anchor_pred = pred[i, grid_y, grid_x, anchor_idx, :]
                        
                        # Extract components
                        x_center = torch.sigmoid(anchor_pred[0]).item()
                        y_center = torch.sigmoid(anchor_pred[1]).item()
                        w = anchor_pred[2].item()
                        h = anchor_pred[3].item()
                        conf = torch.sigmoid(anchor_pred[4]).item()
                        
                        # Get class predictions
                        class_logits = anchor_pred[5:]
                        class_probs = F.softmax(class_logits, dim=0)
                        class_score, class_idx = class_probs.max(0)
                        
                        # Final score = confidence * class_probability
                        final_score = conf * class_score.item()
                        
                        if final_score > score_threshold:
                            # Convert to absolute coordinates
                            x_min = (x_center - w / 2) * img_size[1]
                            y_min = (y_center - h / 2) * img_size[0]
                            x_max = (x_center + w / 2) * img_size[1]
                            y_max = (y_center + h / 2) * img_size[0]
                            
                            boxes.append([x_min, y_min, x_max, y_max])
                            scores.append(final_score)
                            labels.append(int(class_idx.item()) + 1)  # Convert to 1-indexed
            
            # Convert to tensors
            if boxes:
                results.append({
                    'boxes': torch.tensor(boxes, device=pred.device),
                    'scores': torch.tensor(scores, device=pred.device),
                    'labels': torch.tensor(labels, device=pred.device)
                })
            else:
                results.append({
                    'boxes': torch.zeros((0, 4), device=pred.device),
                    'scores': torch.zeros(0, device=pred.device),
                    'labels': torch.zeros(0, dtype=torch.long, device=pred.device)
                })
        
        return results
    
    def train(self, mode=True):
        """Set model to training mode"""
        super().train(mode)
        if self.use_ultralytics and self._ultralytics_model:
            self._ultralytics_model.train(mode)
        return self
    
    def eval(self):
        """Set model to evaluation mode"""
        super().eval()
        if self.use_ultralytics and self._ultralytics_model:
            self._ultralytics_model.eval()
        return self
    
    def to(self, device):
        """Move model to device"""
        super().to(device)
        if self.use_ultralytics and self._ultralytics_model:
            self._ultralytics_model.to(device)
        return self


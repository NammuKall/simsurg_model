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
from torchvision.ops import nms


class YOLOv5Model(nn.Module):
    """
    YOLOv5 model wrapper for SimSurgSkill dataset
    Uses CSPDarkNet backbone with PANet neck and YOLOv5 detection heads
    
    Backend Selection:
    - If ultralytics package is available: Uses ultralytics YOLOv5 for inference
      (training falls back to custom implementation)
    - If ultralytics not available: Uses custom PyTorch implementation
      (ResNet-based backbone with YOLOv5-style detection heads)
    
    Note: Custom implementation is always used for training as ultralytics
    training requires different target formatting. For best results, use
    custom implementation (set pretrained=False or ensure ultralytics unavailable).
    """
    
    def __init__(self, num_classes=2, pretrained=True, model_size='s', input_size=(720, 1280), 
                 loss_weights=None):
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
            input_size: Input image size as (height, width) tuple or single int for square
                        Default: (720, 1280) for SimSurgSkill dataset
                        If single int provided, creates square (int, int)
                        Used for validation - images should be resized to this size
            loss_weights: Dict with loss weights {'coord': float, 'conf': float, 'class': float}
                         Default: {'coord': 0.05, 'conf': 1.0, 'class': 0.5} (YOLOv5 standard)
        """
        super(YOLOv5Model, self).__init__()
        
        self.num_classes = num_classes
        # Normalize input_size to tuple (height, width)
        if isinstance(input_size, (int, float)):
            # Single value -> square input
            self.input_size = (int(input_size), int(input_size))
        elif isinstance(input_size, (tuple, list)) and len(input_size) == 2:
            # Tuple/list -> (height, width)
            self.input_size = (int(input_size[0]), int(input_size[1]))
        else:
            raise ValueError(f"input_size must be int or (height, width) tuple, got {input_size}")
        self.model_size = model_size
        
        # Set loss weights (YOLOv5 standard defaults)
        if loss_weights is None:
            self.loss_weights = {'coord': 0.05, 'conf': 1.0, 'class': 0.5}
        else:
            self.loss_weights = loss_weights
        
        # Try to use ultralytics YOLOv5 if available, otherwise use custom implementation
        if ULTRALYTICS_AVAILABLE:
            self.use_ultralytics = True
            # Initialize ultralytics model in __init__ for better device handling
            try:
                model_name = f'yolov5{self.model_size}.pt'
                self._ultralytics_model = YOLO(model_name)
                # Modify number of classes
                self._ultralytics_model.model.nc = self.num_classes
            except Exception as e:
                # Fallback to custom if ultralytics init fails
                import warnings
                warnings.warn(f"Failed to initialize ultralytics YOLOv5: {e}. Using custom implementation.")
                self.use_ultralytics = False
                self.pretrained = pretrained
                self._build_custom_yolov5()
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
        
        # Number of anchors per scale (YOLOv5 uses 3 anchors per scale)
        # Set this FIRST before calling _make_yolov5_head which needs it
        self.num_anchors = 3
        
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
        
        # Feature fusion for top-down path
        self.fusion1 = nn.Conv2d(backbone_channels[3] + backbone_channels[2], backbone_channels[2], kernel_size=1)
        self.fusion2 = nn.Conv2d(backbone_channels[2] + backbone_channels[1], backbone_channels[1], kernel_size=1)
        
        # Feature fusion for bottom-up path
        # p4_bottom: medium_features (channels[2]) + downsampled_p3 (channels[1]) -> channels[2]
        self.fusion3 = nn.Conv2d(backbone_channels[2] + backbone_channels[1], backbone_channels[2], kernel_size=1)
        # p5_bottom: large_features (channels[3]) + downsampled_p4 (channels[2]) -> channels[3]
        self.fusion4 = nn.Conv2d(backbone_channels[3] + backbone_channels[2], backbone_channels[3], kernel_size=1)
        
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
        # Ensure model is initialized (should be done in __init__, but check for safety)
        if self._ultralytics_model is None:
            try:
                model_name = f'yolov5{self.model_size}.pt'
                self._ultralytics_model = YOLO(model_name)
                self._ultralytics_model.model.nc = self.num_classes
            except Exception as e:
                # Fallback to custom implementation
                import warnings
                warnings.warn(f"Failed to initialize ultralytics YOLOv5: {e}. Using custom implementation.")
                self.use_ultralytics = False
                if not hasattr(self, 'backbone_conv1'):
                    self.pretrained = True
                    self._build_custom_yolov5()
                return self._forward_custom(x, targets)
        
        # Get device from input tensor
        device = x.device if isinstance(x, torch.Tensor) else torch.device('cpu')
        
        # Convert input format
        if isinstance(x, torch.Tensor):
            # Check if images are normalized [0,1] or [0,255]
            # Assume normalized [0,1] if max value <= 1.0
            max_val = x.max().item()
            images = []
            for i in range(x.shape[0]):
                img = x[i].cpu().numpy().transpose(1, 2, 0)
                if max_val <= 1.0:
                    # Images are normalized [0,1], convert to [0,255]
                    img = (img * 255).astype('uint8')
                else:
                    # Images are already [0,255], just convert type
                    img = img.astype('uint8')
                images.append(img)
        else:
            images = x
        
        if targets is not None:
            # Training mode - ultralytics YOLO training requires different format
            # Fall back to custom implementation for training
            import warnings
            warnings.warn(
                "Ultralytics YOLOv5 training mode not fully implemented. "
                "Using custom implementation for training.",
                UserWarning
            )
            self.use_ultralytics = False
            if not hasattr(self, 'backbone_conv1'):
                self.pretrained = True
                self._build_custom_yolov5()
            return self._forward_custom(x, targets)
        else:
            # Inference mode
            results = self._ultralytics_model(images)
            
            # Convert to our format with proper device handling
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
                    'boxes': torch.tensor(boxes, device=device) if boxes else torch.zeros((0, 4), device=device),
                    'scores': torch.tensor(scores, device=device) if scores else torch.zeros(0, device=device),
                    'labels': torch.tensor(labels, device=device, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long, device=device)
                })
            
            return formatted_results
    
    def _forward_custom(self, x, targets=None):
        """Forward pass using custom YOLOv5 implementation"""
        try:
            # Validate input
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(x)}")
            if x.dim() != 4:
                raise ValueError(f"Expected 4D tensor [B, C, H, W], got {x.dim()}D tensor")
            if x.shape[1] != 3:
                raise ValueError(f"Expected 3 channels, got {x.shape[1]} channels")
            
            batch_size = x.shape[0]
            
            # Store input image size for loss computation and postprocessing
            input_img_size = (x.shape[2], x.shape[3])  # (H, W)
            
            # Validate input_size parameter if set (warn if significant mismatch)
            # Note: input_size is a reference size, model can handle different sizes
            if self.input_size[0] > 0 and self.input_size[1] > 0:
                expected_h, expected_w = self.input_size
                actual_h, actual_w = input_img_size
                # Check if aspect ratio matches (allow some tolerance)
                expected_aspect = expected_w / expected_h
                actual_aspect = actual_w / actual_h
                aspect_diff = abs(expected_aspect - actual_aspect)
                
                # Warn if aspect ratio differs significantly (>20%) or size differs significantly
                size_diff_h = abs(actual_h - expected_h) / max(expected_h, 1)
                size_diff_w = abs(actual_w - expected_w) / max(expected_w, 1)
                
                if aspect_diff > 0.2 or size_diff_h > 0.5 or size_diff_w > 0.5:
                    import warnings
                    warnings.warn(
                        f"Input size differs significantly: model reference size is {expected_h}x{expected_w}, "
                        f"got {actual_h}x{actual_w}. Model will adapt, but performance may vary.",
                        UserWarning
                    )
            
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
            
            # Upsample p5 to match medium_features spatial dimensions exactly
            upsampled_p5 = F.interpolate(p5, size=medium_features.shape[2:], mode='nearest', align_corners=None)
            p4 = torch.cat([upsampled_p5, medium_features], dim=1)
            p4 = self.fusion1(p4)
            
            # Upsample p4 to match small_features spatial dimensions exactly
            upsampled_p4 = F.interpolate(p4, size=small_features.shape[2:], mode='nearest', align_corners=None)
            p3 = torch.cat([upsampled_p4, small_features], dim=1)
            p3 = self.fusion2(p3)
            
            # Bottom-up path (for better feature fusion)
            # Downsample p3 to match medium_features spatial dimensions exactly
            downsampled_p3 = F.interpolate(p3, size=medium_features.shape[2:], mode='nearest', align_corners=None)
            p4_bottom = torch.cat([medium_features, downsampled_p3], dim=1)
            p4_bottom = self.fusion3(p4_bottom)  # Fuse to match detection head input channels
            
            # Downsample p4 to match large_features spatial dimensions exactly
            downsampled_p4 = F.interpolate(p4, size=large_features.shape[2:], mode='nearest', align_corners=None)
            p5_bottom = torch.cat([large_features, downsampled_p4], dim=1)
            p5_bottom = self.fusion4(p5_bottom)  # Fuse to match detection head input channels
            
            # Detection at 3 scales
            scale1_out = self.detect_scale1(p5_bottom)  # Large objects
            scale2_out = self.detect_scale2(p4_bottom)  # Medium objects
            scale3_out = self.detect_scale3(p3)  # Small objects
            
            if targets is not None:
                # Training mode: compute loss
                # Validate targets
                if not isinstance(targets, (list, tuple)):
                    raise TypeError(f"Expected list/tuple of targets, got {type(targets)}")
                if len(targets) != batch_size:
                    raise ValueError(f"Targets length {len(targets)} doesn't match batch size {batch_size}")
                
                loss = self.compute_loss([scale1_out, scale2_out, scale3_out], targets, input_img_size)
                return {'loss': loss}
            else:
                # Inference mode: return predictions
                return self.postprocess_predictions([scale1_out, scale2_out, scale3_out], input_img_size)
        except Exception as e:
            raise RuntimeError(f"Error in YOLOv5 custom forward pass: {e}") from e
    
    def compute_loss(self, predictions, targets, img_size):
        """
        Compute YOLOv5 loss - assigns each object to best matching scale and anchor
        
        Args:
            predictions: List of 3 prediction tensors (one per scale)
            targets: List of target dicts
            img_size: Image size (H, W)
        """
        device = predictions[0].device
        
        # Validate batch sizes match
        batch_size = len(targets)
        for pred in predictions:
            if pred.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch: targets has {batch_size}, but prediction has {pred.shape[0]}")
        
        # Handle empty batch
        if batch_size == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Reshape all predictions once
        reshaped_preds = []
        scale_sizes = []  # Store (H, W) for each scale
        for pred in predictions:
            B, _, H, W = pred.shape
            scale_sizes.append((H, W))
            pred_reshaped = pred.view(B, self.num_anchors, 5 + self.num_classes, H, W)
            pred_reshaped = pred_reshaped.permute(0, 3, 4, 1, 2).contiguous()
            reshaped_preds.append(pred_reshaped)
        
        # Collect all losses
        coord_losses = []
        conf_losses = []
        class_losses = []
        
        # Process each image in batch
        for i in range(batch_size):
            gt_boxes = targets[i]['boxes']
            gt_labels = targets[i]['labels']
            
            # Handle empty boxes
            if len(gt_boxes) == 0:
                continue
            
            for box_idx, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
                # Validate box coordinates
                if isinstance(box, torch.Tensor):
                    if box.numel() < 4:
                        continue
                    box_vals = box.cpu().numpy() if box.is_cuda else box.numpy()
                    x_min, y_min, x_max, y_max = box_vals[0], box_vals[1], box_vals[2], box_vals[3]
                else:
                    if len(box) < 4:
                        continue
                    x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
                
                # Validate box is valid
                if x_max <= x_min or y_max <= y_min or x_min < 0 or y_min < 0:
                    continue
                
                # Normalize box coordinates
                x_center = (x_min + x_max) / 2.0 / img_size[1]
                y_center = (y_min + y_max) / 2.0 / img_size[0]
                w = (x_max - x_min) / img_size[1]
                h = (y_max - y_min) / img_size[0]
                
                # Clamp normalized coordinates (now Python floats)
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w = max(0.001, min(1.0, w))
                h = max(0.001, min(1.0, h))
                
                # Convert to tensors for vectorized operations (from Python floats, no warning)
                x_center_t = torch.tensor(x_center, device=device, dtype=torch.float32)
                y_center_t = torch.tensor(y_center, device=device, dtype=torch.float32)
                w_t = torch.tensor(w, device=device, dtype=torch.float32)
                h_t = torch.tensor(h, device=device, dtype=torch.float32)
                
                # Find best scale and anchor for this object
                # Assign to scale based on object size (larger objects to larger scales)
                obj_size = w * h
                if obj_size > 0.25:  # Large objects -> scale 0
                    best_scale = 0
                elif obj_size > 0.0625:  # Medium objects -> scale 1
                    best_scale = 1
                else:  # Small objects -> scale 2
                    best_scale = 2
                
                pred = reshaped_preds[best_scale]
                H, W = scale_sizes[best_scale]
                
                # Find grid cell
                grid_x = int(x_center * W)
                grid_y = int(y_center * H)
                grid_x = max(0, min(W - 1, grid_x))
                grid_y = max(0, min(H - 1, grid_y))
                
                # Find best anchor by comparing predictions
                best_anchor_idx = 0
                best_anchor_loss = float('inf')
                
                for anchor_idx in range(self.num_anchors):
                    anchor_pred = pred[i, grid_y, grid_x, anchor_idx, :]
                    pred_x = torch.sigmoid(anchor_pred[0])
                    pred_y = torch.sigmoid(anchor_pred[1])
                    # Clamp exp() to prevent numerical instability
                    pred_w = torch.clamp(torch.exp(anchor_pred[2]), max=4.0)  # Max 4x image size
                    pred_h = torch.clamp(torch.exp(anchor_pred[3]), max=4.0)
                    
                    # Compute coordinate loss for this anchor
                    anchor_loss = (
                        (pred_x - x_center_t) ** 2 +
                        (pred_y - y_center_t) ** 2 +
                        (pred_w - w_t) ** 2 +
                        (pred_h - h_t) ** 2
                    ).sum()
                    
                    if anchor_loss.item() < best_anchor_loss:
                        best_anchor_loss = anchor_loss.item()
                        best_anchor_idx = anchor_idx
                
                # Use best matching anchor
                anchor_pred = pred[i, grid_y, grid_x, best_anchor_idx, :]
                
                # Extract and compute losses
                pred_x = torch.sigmoid(anchor_pred[0])
                pred_y = torch.sigmoid(anchor_pred[1])
                pred_w = torch.clamp(torch.exp(anchor_pred[2]), max=4.0)
                pred_h = torch.clamp(torch.exp(anchor_pred[3]), max=4.0)
                pred_classes = anchor_pred[5:]
                
                # Coordinate losses
                coord_losses.append((pred_x - x_center_t) ** 2)
                coord_losses.append((pred_y - y_center_t) ** 2)
                coord_losses.append((pred_w - w_t) ** 2)
                coord_losses.append((pred_h - h_t) ** 2)
                
                # Confidence loss (positive example)
                conf_losses.append(F.binary_cross_entropy_with_logits(
                    anchor_pred[4],
                    torch.tensor(1.0, device=device)
                ))
                
                # Classification loss
                label_idx = int(label.item()) - 1
                label_idx = max(0, min(self.num_classes - 1, label_idx))
                class_losses.append(F.cross_entropy(
                    pred_classes.unsqueeze(0),
                    torch.tensor([label_idx], device=device, dtype=torch.long)
                ))
        
        # Combine losses
        if coord_losses:
            total_coord_loss = torch.stack(coord_losses).sum()
            total_conf_loss = torch.stack(conf_losses).sum()
            total_class_loss = torch.stack(class_losses).sum()
            num_objects = len(conf_losses)
            
            total_loss = (
                self.loss_weights['coord'] * total_coord_loss +
                self.loss_weights['conf'] * total_conf_loss +
                self.loss_weights['class'] * total_class_loss
            ) / num_objects
        else:
            # If no objects, return small loss
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss
    
    def postprocess_predictions(self, predictions, img_size, score_threshold=0.25, nms_threshold=0.45):
        """
        Convert raw YOLOv5 predictions to final detections with NMS (vectorized for performance)
        
        Args:
            predictions: List of 3 prediction tensors (one per scale)
            img_size: Image size (H, W)
            score_threshold: Confidence threshold (YOLOv5 default: 0.25)
            nms_threshold: IoU threshold for NMS (YOLOv5 default: 0.45)
        
        Returns:
            List of dicts with 'boxes', 'scores', 'labels' for each image in batch
        """
        # Validate batch sizes match across scales
        batch_size = predictions[0].shape[0]
        device = predictions[0].device
        for pred in predictions[1:]:
            if pred.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch: scale 0 has {batch_size}, but other scale has {pred.shape[0]}")
        
        results = []
        img_h, img_w = img_size[0], img_size[1]
        
        # Process each image in batch
        for i in range(batch_size):
            all_boxes = []
            all_scores = []
            all_labels = []
            
            # Process all scales for this image
            for scale_idx, pred in enumerate(predictions):
                B, _, H, W = pred.shape
                
                # Reshape: [B, anchors, 5+classes, H, W] -> [B, H, W, anchors, 5+classes]
                pred_reshaped = pred.view(B, self.num_anchors, 5 + self.num_classes, H, W)
                pred_reshaped = pred_reshaped.permute(0, 3, 4, 1, 2).contiguous()
                
                # Extract predictions for this image: [H, W, anchors, 5+classes]
                img_pred = pred_reshaped[i]  # [H, W, anchors, 5+classes]
                
                # Vectorized extraction of all components
                # Shape: [H, W, anchors]
                x_center = torch.sigmoid(img_pred[..., 0])
                y_center = torch.sigmoid(img_pred[..., 1])
                w = torch.clamp(torch.exp(img_pred[..., 2]), max=4.0)
                h = torch.clamp(torch.exp(img_pred[..., 3]), max=4.0)
                conf = torch.sigmoid(img_pred[..., 4])
                
                # Class predictions: [H, W, anchors, num_classes]
                class_logits = img_pred[..., 5:]
                class_probs = F.softmax(class_logits, dim=-1)
                class_scores, class_indices = class_probs.max(dim=-1)  # [H, W, anchors]
                
                # Final scores: [H, W, anchors]
                final_scores = conf * class_scores
                
                # Filter by score threshold
                mask = final_scores > score_threshold
                
                if mask.any():
                    # Get indices where mask is True
                    indices = torch.nonzero(mask, as_tuple=False)  # [N, 3] (y, x, anchor)
                    
                    # Extract values at masked positions
                    x_centers = x_center[mask]
                    y_centers = y_center[mask]
                    ws = w[mask]
                    hs = h[mask]
                    scores = final_scores[mask]
                    labels = class_indices[mask] + 1  # Convert to 1-indexed
                    
                    # Convert to absolute coordinates (vectorized)
                    x_min = torch.clamp((x_centers - ws / 2) * img_w, min=0, max=img_w)
                    y_min = torch.clamp((y_centers - hs / 2) * img_h, min=0, max=img_h)
                    x_max = torch.clamp((x_centers + ws / 2) * img_w, min=0, max=img_w)
                    y_max = torch.clamp((y_centers + hs / 2) * img_h, min=0, max=img_h)
                    
                    # Filter valid boxes (x_max > x_min, y_max > y_min)
                    valid_mask = (x_max > x_min) & (y_max > y_min) & (ws > 0) & (hs > 0)
                    
                    if valid_mask.any():
                        boxes = torch.stack([x_min[valid_mask], y_min[valid_mask], 
                                           x_max[valid_mask], y_max[valid_mask]], dim=1)
                        all_boxes.append(boxes)
                        all_scores.append(scores[valid_mask])
                        all_labels.append(labels[valid_mask])
            
            # Combine predictions from all scales
            if all_boxes:
                boxes_tensor = torch.cat(all_boxes, dim=0).to(device)
                scores_tensor = torch.cat(all_scores, dim=0).to(device)
                labels_tensor = torch.cat(all_labels, dim=0).to(device)
                
                # Apply NMS per class
                keep_indices = []
                unique_labels = torch.unique(labels_tensor)
                
                for label in unique_labels:
                    label_mask = labels_tensor == label
                    num_label_detections = label_mask.sum().item()
                    if num_label_detections == 0:
                        continue
                    
                    label_boxes = boxes_tensor[label_mask]
                    label_scores = scores_tensor[label_mask]
                    label_indices = torch.where(label_mask)[0]
                    
                    # Apply NMS (handle edge case of single detection)
                    if num_label_detections == 1:
                        keep = torch.tensor([0], device=device, dtype=torch.long)
                    else:
                        keep = nms(label_boxes, label_scores, nms_threshold)
                    keep_indices.extend(label_indices[keep].tolist())
                
                if keep_indices:
                    keep_indices = torch.tensor(keep_indices, device=device)
                    results.append({
                        'boxes': boxes_tensor[keep_indices],
                        'scores': scores_tensor[keep_indices],
                        'labels': labels_tensor[keep_indices]
                    })
                else:
                    results.append({
                        'boxes': torch.zeros((0, 4), device=device),
                        'scores': torch.zeros(0, device=device),
                        'labels': torch.zeros(0, dtype=torch.long, device=device)
                    })
            else:
                results.append({
                    'boxes': torch.zeros((0, 4), device=device),
                    'scores': torch.zeros(0, device=device),
                    'labels': torch.zeros(0, dtype=torch.long, device=device)
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
        if self.use_ultralytics and self._ultralytics_model is not None:
            try:
                # Move ultralytics model to device
                if hasattr(self._ultralytics_model, 'to'):
                    self._ultralytics_model.to(device)
                elif hasattr(self._ultralytics_model, 'model'):
                    # Try to move underlying model
                    if hasattr(self._ultralytics_model.model, 'to'):
                        self._ultralytics_model.model.to(device)
            except Exception as e:
                import warnings
                warnings.warn(f"Could not move ultralytics model to device {device}: {e}")
        return self


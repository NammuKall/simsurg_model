#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model configuration definitions and management
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for a model"""
    name: str
    class_name: str
    default_params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    supports_pretrained: bool = True
    variants: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# Model configurations
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "EfficientDet": ModelConfig(
        name="EfficientDet",
        class_name="EfficientDetModel",
        default_params={
            "num_classes": 2,
            "num_anchors": 9,
        },
        description="EfficientDet-style model with ResNet50 backbone and feature fusion",
        supports_pretrained=True,
        variants={
            "EfficientDet-Small": {
                "num_anchors": 5,
            },
            "EfficientDet-Large": {
                "num_anchors": 12,
            },
        }
    ),
    "ResNet": ModelConfig(
        name="ResNet",
        class_name="ResNet",
        default_params={
            "num_classes": 2,
            "layers": [2, 2, 2, 2],  # ResNet-18 configuration
        },
        description="ResNet model for classification",
        supports_pretrained=True,
        variants={
            "ResNet-18": {
                "layers": [2, 2, 2, 2],
            },
            "ResNet-34": {
                "layers": [3, 4, 6, 3],
            },
        }
    ),
    "FasterRCNN": ModelConfig(
        name="FasterRCNN",
        class_name="FasterRCNNModel",
        default_params={
            "num_classes": 2,
            "pretrained": True,
            "trainable_backbone_layers": 3,
        },
        description="Faster R-CNN with ResNet50-FPN backbone (torchvision implementation)",
        supports_pretrained=True,
        variants={
            "FasterRCNN-Frozen": {
                "trainable_backbone_layers": 0,
            },
            "FasterRCNN-Full": {
                "trainable_backbone_layers": 5,
            },
        }
    ),
    "YOLOv5": ModelConfig(
        name="YOLOv5",
        class_name="YOLOv5Model",
        default_params={
            "num_classes": 2,
            "pretrained": True,
            "model_size": "s",
            "input_size": 640,
            "loss_weights": {"coord": 0.1, "conf": 1.5, "class": 0.8},  # Optimized loss weights
            "use_focal_loss": False,  # Can enable for hard example mining
            "label_smoothing": 0.0,  # Can enable (e.g., 0.1) for regularization
            "anchor_iou_threshold": 0.5,  # IoU threshold for anchor matching
        },
        description="YOLOv5 model with CSPDarkNet backbone and PANet (optimized hyperparameters)",
        supports_pretrained=True,
        variants={
            "YOLOv5-Nano": {
                "model_size": "n",
                "input_size": 640,
            },
            "YOLOv5-Small": {
                "model_size": "s",
                "input_size": 640,
            },
            "YOLOv5-Medium": {
                "model_size": "m",
                "input_size": 640,
            },
            "YOLOv5-Large": {
                "model_size": "l",
                "input_size": 640,
            },
            "YOLOv5-XLarge": {
                "model_size": "x",
                "input_size": 640,
            },
            "YOLOv5-Focal": {
                "model_size": "s",
                "use_focal_loss": True,  # Enable focal loss for hard examples
            },
            "YOLOv5-Smooth": {
                "model_size": "s",
                "label_smoothing": 0.1,  # Enable label smoothing
            },
        }
    ),
}


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """
    Get configuration for a model
    
    Args:
        model_name: Name of the model
        
    Returns:
        ModelConfig if found, None otherwise
    """
    return MODEL_CONFIGS.get(model_name)


def list_available_models() -> list:
    """Return list of available model names"""
    return list(MODEL_CONFIGS.keys())


def get_model_variants(model_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Get available variants for a model
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of variant names to their configurations
    """
    config = get_model_config(model_name)
    if config:
        return config.variants
    return {}


def get_default_model_name() -> str:
    """Get default model name from environment or return EfficientDet"""
    return os.getenv("MODEL_NAME", "EfficientDet")


def merge_config_with_variant(base_config: Dict[str, Any], variant_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Merge base configuration with variant configuration
    
    Args:
        base_config: Base model configuration
        variant_name: Optional variant name
        
    Returns:
        Merged configuration dictionary
    """
    if variant_name is None:
        return base_config.copy()
    
    # Check if variant exists in any model's variants
    for model_config in MODEL_CONFIGS.values():
        if variant_name in model_config.variants:
            merged = base_config.copy()
            merged.update(model_config.variants[variant_name])
            return merged
    
    return base_config.copy()


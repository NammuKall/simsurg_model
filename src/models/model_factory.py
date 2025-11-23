#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model factory for creating and managing different model architectures.
Supports easy switching between models via configuration.
"""

import os
import torch
import logging
from typing import Optional, Dict, Any

from .model_config import (
    get_model_config,
    get_default_model_name,
    list_available_models,
    merge_config_with_variant,
    MODEL_CONFIGS
)
from .efficientdet import EfficientDetModel
from .resnet import ResNet, ResidualBlock
from .faster_rcnn import FasterRCNNModel
from .yolov5 import YOLOv5Model

logger = logging.getLogger(__name__)

# Model registry mapping names to classes
MODEL_REGISTRY: Dict[str, type] = {
    "EfficientDet": EfficientDetModel,
    "ResNet": ResNet,
    "FasterRCNN": FasterRCNNModel,
    "YOLOv5": YOLOv5Model,
    # Aliases
    "EfficientDetModel": EfficientDetModel,
    "FasterRCNNModel": FasterRCNNModel,
    "YOLOv5Model": YOLOv5Model,
}


def create_model(
    model_name: Optional[str] = None,
    num_classes: int = 2,
    device: Optional[torch.device] = None,
    variant: Optional[str] = None,
    **model_kwargs
) -> torch.nn.Module:
    """
    Create a model instance using factory pattern.
    
    Args:
        model_name: Name of the model to create. If None, reads from MODEL_NAME env var.
                   Options: "EfficientDet", "ResNet", "FasterRCNN"
        num_classes: Number of classes for the model (default: 2)
        device: Device to move model to (optional)
        variant: Optional variant name (e.g., "EfficientDet-Small", "FasterRCNN-Frozen")
        **model_kwargs: Additional model-specific arguments that override defaults
    
    Returns:
        Initialized model instance
    
    Raises:
        ValueError: If model_name is not in registry or configuration is invalid
    
    Example:
        >>> # Create EfficientDet model
        >>> model = create_model("EfficientDet", num_classes=2, device=device)
        >>> 
        >>> # Create Faster R-CNN with frozen backbone
        >>> model = create_model("FasterRCNN", variant="FasterRCNN-Frozen")
        >>>
        >>> # Create model from environment variable
        >>> # Set MODEL_NAME=FasterRCNN in .env
        >>> model = create_model(num_classes=2, device=device)
    """
    # Get model name from parameter or environment
    if model_name is None:
        model_name = get_default_model_name()
    
    model_name = model_name.strip()
    
    # Validate model name
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(list_available_models())
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available models: {available}"
        )
    
    # Get model class
    model_class = MODEL_REGISTRY[model_name]
    
    # Get model configuration
    config = get_model_config(model_name)
    if config is None:
        # Fallback: try to find config by checking aliases
        for key, value in MODEL_CONFIGS.items():
            if value.class_name == model_class.__name__:
                config = value
                break
    
    # Build parameters dictionary
    params = {}
    
    # Start with default parameters from config
    if config:
        params.update(config.default_params)
    
    # Apply variant if specified
    if variant:
        variant_config = merge_config_with_variant(params, variant)
        params.update(variant_config)
    
    # Override with explicit parameters
    params.update(model_kwargs)
    
    # Ensure num_classes is set (can be overridden)
    if 'num_classes' not in params:
        params['num_classes'] = num_classes
    else:
        # Use provided num_classes if explicitly set
        params['num_classes'] = num_classes
    
    # Special handling for ResNet (requires block and layers)
    if model_class == ResNet:
        if 'block' not in params:
            params['block'] = ResidualBlock
        if 'layers' not in params:
            params['layers'] = [2, 2, 2, 2]  # Default ResNet-18
    
    # Create model instance
    logger.info(f"Creating {model_name} model with {params['num_classes']} classes")
    if variant:
        logger.info(f"Using variant: {variant}")
    logger.debug(f"Model parameters: {params}")
    
    try:
        model = model_class(**params)
    except Exception as e:
        logger.error(f"Failed to create {model_name} model: {e}")
        raise ValueError(f"Failed to create {model_name} model: {e}") from e
    
    # Move to device if specified
    if device is not None:
        model = model.to(device)
        logger.info(f"Model moved to device: {device}")
    
    return model


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model information
        
    Raises:
        ValueError: If model_name is not found
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_class = MODEL_REGISTRY[model_name]
    config = get_model_config(model_name)
    
    info = {
        "name": model_name,
        "class": model_class.__name__,
        "module": model_class.__module__,
        "doc": model_class.__doc__,
    }
    
    if config:
        info.update({
            "description": config.description,
            "supports_pretrained": config.supports_pretrained,
            "default_params": config.default_params,
            "variants": list(config.variants.keys()) if config.variants else [],
        })
    
    return info


def list_models() -> list:
    """
    List all available models.
    
    Returns:
        List of available model names
    """
    return list_available_models()


def get_model_summary() -> str:
    """
    Get a formatted summary of all available models.
    
    Returns:
        Formatted string with model information
    """
    lines = ["Available Models:", "=" * 50]
    
    for model_name in list_available_models():
        try:
            info = get_model_info(model_name)
            lines.append(f"\n{model_name}:")
            lines.append(f"  Description: {info.get('description', 'N/A')}")
            lines.append(f"  Class: {info.get('class', 'N/A')}")
            lines.append(f"  Supports Pretrained: {info.get('supports_pretrained', False)}")
            if info.get('variants'):
                lines.append(f"  Variants: {', '.join(info['variants'])}")
        except Exception as e:
            lines.append(f"\n{model_name}: Error getting info - {e}")
    
    return "\n".join(lines)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward compatibility wrapper for models.
All models have been moved to src/models/ directory.
This file maintains backward compatibility by re-exporting from the new location.
"""

# Re-export all models from the new models package
from src.models import (
    EfficientDetModel,
    ResNet,
    ResidualBlock,
    FasterRCNNModel,
    YOLOv5Model,
)

# Also export the factory functions for convenience
from src.models.model_factory import (
    create_model,
    get_model_info,
    list_models,
    get_model_summary,
)

__all__ = [
    'EfficientDetModel',
    'ResNet',
    'ResidualBlock',
    'FasterRCNNModel',
    'YOLOv5Model',
    'create_model',
    'get_model_info',
    'list_models',
    'get_model_summary',
]

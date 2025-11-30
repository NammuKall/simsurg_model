"""
Data handling module for SimSurgSkill dataset processing.

This module contains all data-related functionality including:
- Data extraction from videos
- Data wrangling and validation
- COCO format conversion
- Data loading for training
- Plotting and visualization
"""

from .coco_data_loader import get_coco_data_loaders
from .coco_json import COCOJSONGenerator
from .coco_converter import convert_to_coco_format
from .data_extractor import VideoFrameExtractor
from .data_wrangler import DataWrangler
from .plots import PlotGenerator

__all__ = [
    'get_coco_data_loaders',
    'COCOJSONGenerator',
    'convert_to_coco_format',
    'VideoFrameExtractor',
    'DataWrangler',
    'PlotGenerator',
]


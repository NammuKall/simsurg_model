#data_wranger.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Wrangler - Validate and organize annotations with corresponding images
Ensures data integrity before COCO conversion
FIXED: Only extracts frame_id=1 annotations to match fps=1 extracted images
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import logging
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataWrangler:
    """Wrangle and validate dataset annotations with images"""
    
    def __init__(self, base_dir):
        """
        Initialize the wrangler
        
        Args:
            base_dir (str): Base directory containing the dataset
        """
        self.base_dir = Path(base_dir)
        self.wrangled_data = {
            'train_v1': [],
            'train_v2': [],
            'test': []
        }
        self.stats = {
            'train_v1': defaultdict(int),
            'train_v2': defaultdict(int),
            'test': defaultdict(int)
        }
        
    def parse_coordinate_string(self, coord_str):
        """
        Parse coordinate string from annotation
        
        Args:
            coord_str (str): Coordinate string like '{"h": 108, "w": 77, "x": 625, "y": 420}'
            
        Returns:
            dict: Parsed coordinates
        """
        try:
            return json.loads(coord_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse coordinate string: {coord_str}")
            raise e
    
    def validate_annotation(self, annotation, image_width=1280, image_height=720):
        """
        Validate a single annotation
        
        Args:
            annotation (dict): Annotation dictionary
            image_width (int): Image width for bounds checking
            image_height (int): Image height for bounds checking
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Parse coordinates
            coords = self.parse_coordinate_string(annotation['coordinate'])
            
            # Check required fields
            required_fields = ['obj_class', 'coordinate', 'orientation', 'frame_id']
            for field in required_fields:
                if field not in annotation:
                    logger.warning(f"Missing field: {field}")
                    return False
            
            # Validate coordinate values
            x, y, w, h = coords['x'], coords['y'], coords['w'], coords['h']
            
            # Check if coordinates are within image bounds
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                logger.warning(f"Invalid coordinates: x={x}, y={y}, w={w}, h={h}")
                return False
            
            if x + w > image_width * 1.1 or y + h > image_height * 1.1:
                # Allow 10% tolerance for slight overflows
                logger.debug(f"Coordinates slightly out of bounds: x+w={x+w}, y+h={y+h}")
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def wrangle_split(self, split_name):
        """
        Wrangle data for a single split
        FIXED: Only extracts annotations where frame_id == 1.0
        
        Args:
            split_name (str): Name of the split (train_v1, train_v2, or test)
            
        Returns:
            list: Wrangled data records
        """
        logger.info(f"\nWrangling {split_name} split...")
        
        # Define paths
        image_dir = self.base_dir / split_name / 'videos' / 'fps1'
        annotation_dir = self.base_dir / split_name / 'annotations' / 'bounding_box_gt'
        
        # Check if directories exist
        if not image_dir.exists():
            logger.error(f"Image directory not found: {image_dir}")
            return []
        
        if not annotation_dir.exists():
            logger.error(f"Annotation directory not found: {annotation_dir}")
            return []
        
        # Get all image files
        image_files = list(image_dir.glob('*.jpeg')) + list(image_dir.glob('*.jpg'))
        logger.info(f"Found {len(image_files)} images")
        
        # Get all annotation files
        annotation_files = list(annotation_dir.glob('*.json'))
        logger.info(f"Found {len(annotation_files)} annotation files")
        
        # Create lookup dictionaries
        images_dict = {img.stem: img for img in image_files}
        annotations_dict = {ann.stem: ann for ann in annotation_files}
        
        wrangled_records = []
        
        # Match images with annotations
        for img_name, img_path in images_dict.items():
            if img_name in annotations_dict:
                ann_path = annotations_dict[img_name]
                
                try:
                    # Load annotation file
                    with open(ann_path, 'r') as f:
                        all_annotations = json.load(f)
                    
                    # Get image dimensions
                    img = cv2.imread(str(img_path))
                    if img is None:
                        logger.warning(f"Could not read image: {img_path}")
                        self.stats[split_name]['images_unreadable'] += 1
                        continue
                    
                    img_height, img_width = img.shape[:2]
                    
                    # CRITICAL FIX: Only extract annotations for frame_id == 1
                    # Since we extracted images at fps=1, each image is the FIRST frame
                    # The JSON file contains annotations for ALL frames (1-126+)
                    # We only want annotations for frame_id == 1.0
                    
                    valid_annotations = []
                    frame_1_found = False
                    
                    for ann_id, ann in all_annotations.items():
                        # Only process annotations for frame_id == 1
                        if ann.get('frame_id') == 1.0:
                            frame_1_found = True
                            if self.validate_annotation(ann, img_width, img_height):
                                # Parse coordinates
                                coords = self.parse_coordinate_string(ann['coordinate'])
                                
                                # Create standardized annotation
                                valid_annotations.append({
                                    'annotation_id': ann_id,
                                    'obj_class': ann['obj_class'],
                                    'bbox': [coords['x'], coords['y'], coords['w'], coords['h']],
                                    'orientation': ann['orientation'],
                                    'frame_id': ann['frame_id'],
                                    'case_id': ann.get('case_id', None)
                                })
                            else:
                                self.stats[split_name]['invalid_annotations'] += 1
                    
                    if not frame_1_found:
                        logger.warning(f"No frame_id=1 annotations found for {img_name}")
                        self.stats[split_name]['no_frame_1'] += 1
                    
                    if valid_annotations:
                        # Create wrangled record
                        record = {
                            'image_path': str(img_path),
                            'image_name': img_path.name,
                            'annotation_path': str(ann_path),
                            'image_width': img_width,
                            'image_height': img_height,
                            'annotations': valid_annotations,
                            'split': split_name
                        }
                        wrangled_records.append(record)
                        self.stats[split_name]['valid_pairs'] += 1
                        self.stats[split_name]['total_annotations'] += len(valid_annotations)
                    else:
                        logger.warning(f"No valid annotations for {img_name} at frame_id=1")
                        self.stats[split_name]['no_valid_annotations'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {img_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    self.stats[split_name]['processing_errors'] += 1
            else:
                logger.warning(f"No annotation found for image: {img_name}")
                self.stats[split_name]['missing_annotations'] += 1
        
        # Check for annotations without images
        for ann_name in annotations_dict.keys():
            if ann_name not in images_dict:
                logger.warning(f"No image found for annotation: {ann_name}")
                self.stats[split_name]['missing_images'] += 1
        
        logger.info(f"Wrangled {len(wrangled_records)} valid image-annotation pairs")
        
        return wrangled_records
    
    def wrangle_all_splits(self, splits=['train_v1', 'train_v2', 'test']):
        """
        Wrangle all dataset splits
        
        Args:
            splits (list): List of split names to process
            
        Returns:
            dict: Wrangled data for all splits
        """
        for split in splits:
            self.wrangled_data[split] = self.wrangle_split(split)
        
        # Print summary
        self.print_summary()
        
        return self.wrangled_data
    
    def print_summary(self):
        """Print wrangling summary statistics"""
        logger.info("\n" + "="*60)
        logger.info("DATA WRANGLING SUMMARY")
        logger.info("="*60)
        
        for split, data in self.wrangled_data.items():
            if not data:
                continue
            
            logger.info(f"\n{split.upper()}:")
            logger.info(f"  Valid pairs: {self.stats[split]['valid_pairs']}")
            logger.info(f"  Total annotations: {self.stats[split]['total_annotations']}")
            
            if self.stats[split]['valid_pairs'] > 0:
                avg_ann = self.stats[split]['total_annotations'] / self.stats[split]['valid_pairs']
                logger.info(f"  Avg annotations/image: {avg_ann:.1f}")
            
            if self.stats[split]['missing_annotations'] > 0:
                logger.info(f"  ⚠️  Missing annotations: {self.stats[split]['missing_annotations']}")
            if self.stats[split]['missing_images'] > 0:
                logger.info(f"  ⚠️  Missing images: {self.stats[split]['missing_images']}")
            if self.stats[split]['invalid_annotations'] > 0:
                logger.info(f"  ⚠️  Invalid annotations: {self.stats[split]['invalid_annotations']}")
            if self.stats[split]['processing_errors'] > 0:
                logger.info(f"  ❌ Processing errors: {self.stats[split]['processing_errors']}")
        
        # Overall statistics
        total_pairs = sum(len(data) for data in self.wrangled_data.values())
        total_annotations = sum(self.stats[split]['total_annotations'] for split in self.wrangled_data.keys())
        
        logger.info(f"\nOVERALL:")
        logger.info(f"  Total valid pairs: {total_pairs}")
        logger.info(f"  Total annotations: {total_annotations}")
        if total_pairs > 0:
            logger.info(f"  Average annotations per image: {total_annotations/total_pairs:.1f}")
            logger.info(f"  Expected: ~3 annotations per image (1 needle + 2 needle drivers)")
    
    def save_wrangled_data(self, output_file='wrangled_data.json'):
        """
        Save wrangled data to JSON file
        
        Args:
            output_file (str): Output file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.wrangled_data, f, indent=2)
        
        logger.info(f"\nWrangled data saved to: {output_path}")
    
    def get_class_distribution(self):
        """
        Get class distribution across all splits
        
        Returns:
            dict: Class counts for each split
        """
        distribution = {}
        
        for split, data in self.wrangled_data.items():
            class_counts = defaultdict(int)
            
            for record in data:
                for ann in record['annotations']:
                    class_counts[ann['obj_class']] += 1
            
            distribution[split] = dict(class_counts)
        
        return distribution


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Wrangle and validate dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory containing the dataset')
    parser.add_argument('--output', type=str, default='wrangled_data.json',
                        help='Output file for wrangled data')
    parser.add_argument('--splits', nargs='+', default=['train_v1', 'train_v2', 'test'],
                        help='Dataset splits to process')
    
    args = parser.parse_args()
    
    # Create wrangler
    wrangler = DataWrangler(args.data_dir)
    
    # Wrangle all splits
    wrangled_data = wrangler.wrangle_all_splits(splits=args.splits)
    
    # Save results
    wrangler.save_wrangled_data(args.output)
    
    # Print class distribution
    distribution = wrangler.get_class_distribution()
    logger.info("\nCLASS DISTRIBUTION:")
    for split, classes in distribution.items():
        logger.info(f"  {split}: {classes}")
    
    return wrangled_data


if __name__ == "__main__":
    main()

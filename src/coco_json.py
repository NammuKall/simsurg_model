#coco_json.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO JSON Generator - Create COCO format annotations from wrangled data
FIXED: Properly splits data 70/15/15 regardless of original split sizes
"""

import os
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class COCOJSONGenerator:
    """Generate COCO format JSON from wrangled data"""
    
    def __init__(self, output_dir='data/coco_format'):
        """
        Initialize the generator
        
        Args:
            output_dir (str): Output directory for COCO format data
        """
        self.output_dir = Path(output_dir)
        self.categories = [
            {"id": 1, "name": "needle", "supercategory": "surgical_tool"},
            {"id": 2, "name": "needle_driver", "supercategory": "surgical_tool"}
        ]
        self.class_to_id = {
            'needle': 1,
            'needle driver': 2,
            'needle_driver': 2  # Handle both formats
        }
    
    def create_directory_structure(self):
        """Create COCO format directory structure"""
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        
        (self.output_dir / 'annotations').mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created COCO directory structure at: {self.output_dir}")
    
    def create_info_section(self):
        """Create COCO info section"""
        return {
            "description": "SimSurgSkill 2021 Dataset - Surgical Skill Assessment",
            "url": "https://github.com/your-repo",
            "version": "1.0",
            "year": 2021,
            "contributor": "SimSurgSkill Team",
            "date_created": datetime.now().strftime("%Y/%m/%d")
        }
    
    def create_licenses_section(self):
        """Create COCO licenses section"""
        return [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ]
    
    def split_data(self, wrangled_data, train_ratio=0.7, val_ratio=0.15, random_state=42):
        """
        Split wrangled data into train/val/test sets
        FIXED: Combines all data and does fresh 70/15/15 split
        
        Args:
            wrangled_data (dict): Wrangled data from DataWrangler
            train_ratio (float): Ratio for training set
            val_ratio (float): Ratio for validation set
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Split data for train, val, test
        """
        # Combine ALL data from all original splits
        all_train_data = wrangled_data.get('train_v1', []) + wrangled_data.get('train_v2', [])
        test_data = wrangled_data.get('test', [])
        
        # Combine everything for a fresh split
        total_data = all_train_data + test_data
        
        logger.info(f"\nSplitting data:")
        logger.info(f"  train_v1: {len(wrangled_data.get('train_v1', []))} images")
        logger.info(f"  train_v2: {len(wrangled_data.get('train_v2', []))} images")
        logger.info(f"  test: {len(wrangled_data.get('test', []))} images")
        logger.info(f"  Total available: {len(total_data)} images")
        
        if len(total_data) == 0:
            logger.error("No data available for splitting!")
            return {'train': [], 'val': [], 'test': []}
        
        # Calculate test size
        test_ratio = 1.0 - train_ratio - val_ratio
        
        logger.info(f"\nPerforming fresh random split:")
        logger.info(f"  Target: Train={train_ratio*100:.0f}%, Val={val_ratio*100:.0f}%, Test={test_ratio*100:.0f}%")
        
        # First split: separate out test set
        train_val_data, test_data = train_test_split(
            total_data, 
            test_size=test_ratio,
            random_state=random_state
        )
        
        # Second split: separate train and val from remaining data
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_size_adjusted,
            random_state=random_state
        )
        
        logger.info(f"\nFinal split:")
        logger.info(f"  Train: {len(train_data)} images ({len(train_data)/len(total_data)*100:.1f}%)")
        logger.info(f"  Val:   {len(val_data)} images ({len(val_data)/len(total_data)*100:.1f}%)")
        logger.info(f"  Test:  {len(test_data)} images ({len(test_data)/len(total_data)*100:.1f}%)")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def create_coco_dataset(self, data_records, split_name):
        """
        Create COCO format dataset for a split
        
        Args:
            data_records (list): List of wrangled data records
            split_name (str): Name of the split (train, val, test)
            
        Returns:
            dict: COCO format dataset
        """
        logger.info(f"\nCreating COCO dataset for {split_name}...")
        
        # Initialize COCO structure
        coco_data = {
            "info": self.create_info_section(),
            "licenses": self.create_licenses_section(),
            "images": [],
            "annotations": [],
            "categories": self.categories
        }
        
        image_id = 0
        annotation_id = 0
        
        # Output directory for images
        output_img_dir = self.output_dir / split_name / 'images'
        
        for record in data_records:
            # Copy image to output directory
            src_img_path = Path(record['image_path'])
            dst_img_path = output_img_dir / record['image_name']
            
            try:
                shutil.copy(src_img_path, dst_img_path)
            except Exception as e:
                logger.error(f"Failed to copy image {src_img_path}: {e}")
                continue
            
            # Add image info
            image_info = {
                "id": image_id,
                "file_name": record['image_name'],
                "width": record['image_width'],
                "height": record['image_height'],
                "license": 1,
                "date_captured": ""
            }
            coco_data["images"].append(image_info)
            
            # Add annotations
            for ann in record['annotations']:
                # Get category ID
                category_id = self.class_to_id.get(ann['obj_class'], 1)
                
                # Calculate area
                bbox = ann['bbox']
                area = bbox[2] * bbox[3]  # width * height
                
                # Create COCO annotation
                coco_annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,  # [x, y, width, height]
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": []  # Empty for bbox-only annotations
                }
                
                coco_data["annotations"].append(coco_annotation)
                annotation_id += 1
            
            image_id += 1
        
        logger.info(f"  Created {len(coco_data['images'])} images")
        logger.info(f"  Created {len(coco_data['annotations'])} annotations")
        if len(coco_data['images']) > 0:
            avg_ann = len(coco_data['annotations']) / len(coco_data['images'])
            logger.info(f"  Average {avg_ann:.1f} annotations per image")
        
        return coco_data
    
    def save_coco_json(self, coco_data, split_name):
        """
        Save COCO format JSON file
        
        Args:
            coco_data (dict): COCO format dataset
            split_name (str): Name of the split
        """
        output_file = self.output_dir / 'annotations' / f'instances_{split_name}.json'
        
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        logger.info(f"  Saved to: {output_file}")
    
    def generate_from_wrangled_data(self, wrangled_data, train_ratio=0.7, val_ratio=0.15):
        """
        Generate complete COCO format dataset from wrangled data
        
        Args:
            wrangled_data (dict): Wrangled data from DataWrangler
            train_ratio (float): Ratio for training set
            val_ratio (float): Ratio for validation set
            
        Returns:
            dict: Paths to created COCO datasets
        """
        # Create directory structure
        self.create_directory_structure()
        
        # Split data
        split_data = self.split_data(wrangled_data, train_ratio, val_ratio)
        
        # Create COCO datasets for each split
        coco_datasets = {}
        for split_name, data_records in split_data.items():
            if not data_records:
                logger.warning(f"No data for {split_name} split")
                continue
            
            # Create COCO dataset
            coco_data = self.create_coco_dataset(data_records, split_name)
            
            # Save to JSON
            self.save_coco_json(coco_data, split_name)
            
            coco_datasets[split_name] = coco_data
        
        # Print summary
        self.print_summary(coco_datasets)
        
        # Return paths
        return {
            'train_dir': str(self.output_dir / 'train' / 'images'),
            'val_dir': str(self.output_dir / 'val' / 'images'),
            'test_dir': str(self.output_dir / 'test' / 'images'),
            'train_ann': str(self.output_dir / 'annotations' / 'instances_train.json'),
            'val_ann': str(self.output_dir / 'annotations' / 'instances_val.json'),
            'test_ann': str(self.output_dir / 'annotations' / 'instances_test.json'),
            'output_dir': str(self.output_dir)
        }
    
    def print_summary(self, coco_datasets):
        """Print COCO generation summary"""
        logger.info("\n" + "="*60)
        logger.info("COCO JSON GENERATION SUMMARY")
        logger.info("="*60)
        
        total_images = 0
        total_annotations = 0
        
        for split_name, coco_data in coco_datasets.items():
            n_images = len(coco_data['images'])
            n_annotations = len(coco_data['annotations'])
            
            total_images += n_images
            total_annotations += n_annotations
            
            logger.info(f"\n{split_name.upper()}:")
            logger.info(f"  Images: {n_images}")
            logger.info(f"  Annotations: {n_annotations}")
            if n_images > 0:
                avg_ann = n_annotations / n_images
                logger.info(f"  Avg annotations/image: {avg_ann:.1f}")
                if avg_ann < 2 or avg_ann > 5:
                    logger.warning(f"  ⚠️  Expected ~3 annotations/image, got {avg_ann:.1f}")
            
            # Class distribution
            class_counts = {}
            for ann in coco_data['annotations']:
                cat_id = ann['category_id']
                cat_name = next(c['name'] for c in self.categories if c['id'] == cat_id)
                class_counts[cat_name] = class_counts.get(cat_name, 0) + 1
            
            logger.info(f"  Class distribution: {class_counts}")
        
        logger.info(f"\nTOTAL:")
        logger.info(f"  Images: {total_images}")
        logger.info(f"  Annotations: {total_annotations}")
        if total_images > 0:
            logger.info(f"  Overall avg: {total_annotations/total_images:.1f} annotations/image")


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate COCO format JSON')
    parser.add_argument('--wrangled_data', type=str, required=True,
                        help='Path to wrangled data JSON file')
    parser.add_argument('--output_dir', type=str, default='data/coco_format',
                        help='Output directory for COCO format')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    
    args = parser.parse_args()
    
    # Load wrangled data
    logger.info(f"Loading wrangled data from {args.wrangled_data}...")
    with open(args.wrangled_data, 'r') as f:
        wrangled_data = json.load(f)
    
    # Create generator
    generator = COCOJSONGenerator(output_dir=args.output_dir)
    
    # Generate COCO format
    coco_paths = generator.generate_from_wrangled_data(
        wrangled_data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    logger.info("\n✅ COCO format generation complete!")
    logger.info(f"Output directory: {coco_paths['output_dir']}")
    
    return coco_paths


if __name__ == "__main__":
    main()

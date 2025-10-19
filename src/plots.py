#plots.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots Generator - Create comprehensive visualizations from COCO data
"""

import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import cv2
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


class PlotGenerator:
    """Generate visualizations from COCO format data"""
    
    def __init__(self, coco_paths, output_dir='plots'):
        """
        Initialize plot generator
        
        Args:
            coco_paths (dict): Dictionary with paths to COCO datasets
            output_dir (str): Directory to save plots
        """
        self.coco_paths = coco_paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load COCO data
        self.coco_data = {}
        for split in ['train', 'val', 'test']:
            ann_path = coco_paths.get(f'{split}_ann')
            if ann_path and Path(ann_path).exists():
                with open(ann_path, 'r') as f:
                    self.coco_data[split] = json.load(f)
                logger.info(f"Loaded {split} annotations: {len(self.coco_data[split]['images'])} images")
        
        # Color palette
        self.colors = {
            'needle': '#e74c3c',
            'needle_driver': '#3498db',
            'train': '#2ecc71',
            'val': '#f39c12',
            'test': '#9b59b6'
        }
    
    def plot_dataset_overview(self):
        """Create dataset overview plot"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dataset Overview', fontsize=18, fontweight='bold')
        
        # 1. Images per split
        ax = axes[0, 0]
        splits = []
        counts = []
        colors = []
        
        for split in ['train', 'val', 'test']:
            if split in self.coco_data:
                splits.append(split.capitalize())
                counts.append(len(self.coco_data[split]['images']))
                colors.append(self.colors[split])
        
        bars = ax.bar(splits, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_title('Images per Split', fontweight='bold', fontsize=14)
        ax.set_ylabel('Number of Images', fontsize=12)
        ax.set_xlabel('Split', fontsize=12)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # 2. Annotations per split
        ax = axes[0, 1]
        splits = []
        counts = []
        colors = []
        
        for split in ['train', 'val', 'test']:
            if split in self.coco_data:
                splits.append(split.capitalize())
                counts.append(len(self.coco_data[split]['annotations']))
                colors.append(self.colors[split])
        
        bars = ax.bar(splits, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_title('Annotations per Split', fontweight='bold', fontsize=14)
        ax.set_ylabel('Number of Annotations', fontsize=12)
        ax.set_xlabel('Split', fontsize=12)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # 3. Class distribution (combined)
        ax = axes[1, 0]
        class_counts = defaultdict(int)
        
        for split in self.coco_data.values():
            for ann in split['annotations']:
                cat_id = ann['category_id']
                cat_name = next(c['name'] for c in split['categories'] if c['id'] == cat_id)
                class_counts[cat_name] += 1
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        colors_list = [self.colors.get(c, '#95a5a6') for c in classes]
        
        bars = ax.bar(classes, counts, color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_title('Overall Class Distribution', fontweight='bold', fontsize=14)
        ax.set_ylabel('Number of Annotations', fontsize=12)
        ax.set_xlabel('Class', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # 4. Annotations per image distribution
        ax = axes[1, 1]
        all_ann_counts = []
        
        for split_data in self.coco_data.values():
            img_ann_counts = defaultdict(int)
            for ann in split_data['annotations']:
                img_ann_counts[ann['image_id']] += 1
            all_ann_counts.extend(img_ann_counts.values())
        
        ax.hist(all_ann_counts, bins=range(0, max(all_ann_counts)+2), 
               color='#34495e', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_title('Annotations per Image Distribution', fontweight='bold', fontsize=14)
        ax.set_xlabel('Number of Annotations', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.axvline(np.mean(all_ann_counts), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(all_ann_counts):.2f}')
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        save_path = self.output_dir / 'dataset_overview.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        plt.close()
    
    def plot_class_distribution_by_split(self):
        """Plot class distribution for each split"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Class Distribution by Split', fontsize=18, fontweight='bold')
        
        for idx, split in enumerate(['train', 'val', 'test']):
            if split not in self.coco_data:
                continue
            
            ax = axes[idx]
            split_data = self.coco_data[split]
            
            class_counts = defaultdict(int)
            for ann in split_data['annotations']:
                cat_id = ann['category_id']
                cat_name = next(c['name'] for c in split_data['categories'] if c['id'] == cat_id)
                class_counts[cat_name] += 1
            
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            colors_list = [self.colors.get(c, '#95a5a6') for c in classes]
            
            bars = ax.bar(classes, counts, color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)
            ax.set_title(f'{split.capitalize()}', fontweight='bold', fontsize=14)
            ax.set_ylabel('Count', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(count)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        save_path = self.output_dir / 'class_distribution_by_split.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        plt.close()
    
    def plot_bbox_statistics(self):
        """Plot bounding box statistics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Bounding Box Statistics', fontsize=18, fontweight='bold')
        
        # Collect all bbox data
        all_widths = defaultdict(list)
        all_heights = defaultdict(list)
        all_areas = defaultdict(list)
        
        for split_data in self.coco_data.values():
            for ann in split_data['annotations']:
                cat_id = ann['category_id']
                cat_name = next(c['name'] for c in split_data['categories'] if c['id'] == cat_id)
                bbox = ann['bbox']
                
                all_widths[cat_name].append(bbox[2])
                all_heights[cat_name].append(bbox[3])
                all_areas[cat_name].append(bbox[2] * bbox[3])
        
        # Width distribution
        ax = axes[0, 0]
        for class_name, widths in all_widths.items():
            ax.hist(widths, bins=30, alpha=0.6, label=class_name, 
                   color=self.colors.get(class_name, '#95a5a6'), edgecolor='black')
        ax.set_title('Width Distribution', fontweight='bold', fontsize=14)
        ax.set_xlabel('Width (pixels)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Height distribution
        ax = axes[0, 1]
        for class_name, heights in all_heights.items():
            ax.hist(heights, bins=30, alpha=0.6, label=class_name,
                   color=self.colors.get(class_name, '#95a5a6'), edgecolor='black')
        ax.set_title('Height Distribution', fontweight='bold', fontsize=14)
        ax.set_xlabel('Height (pixels)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Area distribution
        ax = axes[0, 2]
        for class_name, areas in all_areas.items():
            ax.hist(areas, bins=30, alpha=0.6, label=class_name,
                   color=self.colors.get(class_name, '#95a5a6'), edgecolor='black')
        ax.set_title('Area Distribution', fontweight='bold', fontsize=14)
        ax.set_xlabel('Area (pixels²)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Box plots for width
        ax = axes[1, 0]
        data_to_plot = [all_widths[c] for c in all_widths.keys()]
        bp = ax.boxplot(data_to_plot, labels=list(all_widths.keys()), patch_artist=True)
        for patch, class_name in zip(bp['boxes'], all_widths.keys()):
            patch.set_facecolor(self.colors.get(class_name, '#95a5a6'))
        ax.set_title('Width Box Plot', fontweight='bold', fontsize=14)
        ax.set_ylabel('Width (pixels)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Box plots for height
        ax = axes[1, 1]
        data_to_plot = [all_heights[c] for c in all_heights.keys()]
        bp = ax.boxplot(data_to_plot, labels=list(all_heights.keys()), patch_artist=True)
        for patch, class_name in zip(bp['boxes'], all_heights.keys()):
            patch.set_facecolor(self.colors.get(class_name, '#95a5a6'))
        ax.set_title('Height Box Plot', fontweight='bold', fontsize=14)
        ax.set_ylabel('Height (pixels)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Scatter: width vs height
        ax = axes[1, 2]
        for class_name in all_widths.keys():
            ax.scatter(all_widths[class_name], all_heights[class_name], 
                      alpha=0.5, label=class_name, s=20,
                      color=self.colors.get(class_name, '#95a5a6'))
        ax.set_title('Width vs Height', fontweight='bold', fontsize=14)
        ax.set_xlabel('Width (pixels)', fontsize=12)
        ax.set_ylabel('Height (pixels)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'bbox_statistics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        plt.close()
    
    def visualize_sample_annotations(self, num_samples=6):
        """Visualize sample images with annotations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Sample Annotated Images', fontsize=18, fontweight='bold')
        axes = axes.flatten()
        
        # Get sample images from training set
        if 'train' in self.coco_data:
            train_data = self.coco_data['train']
            train_dir = Path(self.coco_paths['train_dir'])
            
            # Randomly sample images
            sample_indices = random.sample(range(len(train_data['images'])), 
                                         min(num_samples, len(train_data['images'])))
            
            for idx, img_idx in enumerate(sample_indices):
                ax = axes[idx]
                img_info = train_data['images'][img_idx]
                img_path = train_dir / img_info['file_name']
                
                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Draw bounding boxes
                img_anns = [ann for ann in train_data['annotations'] 
                           if ann['image_id'] == img_info['id']]
                
                for ann in img_anns:
                    bbox = ann['bbox']
                    cat_id = ann['category_id']
                    cat_name = next(c['name'] for c in train_data['categories'] if c['id'] == cat_id)
                    
                    x, y, w, h = [int(v) for v in bbox]
                    color = tuple(int(self.colors.get(cat_name, '#95a5a6')[i:i+2], 16) for i in (1, 3, 5))
                    
                    cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
                    cv2.putText(img, cat_name, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                ax.imshow(img)
                ax.set_title(f'Image: {img_info["file_name"][:20]}...', fontsize=10)
                ax.axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / 'sample_annotations.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all plots"""
        logger.info("\n" + "="*60)
        logger.info("GENERATING PLOTS")
        logger.info("="*60)
        
        self.plot_dataset_overview()
        self.plot_class_distribution_by_split()
        self.plot_bbox_statistics()
        self.visualize_sample_annotations()
        
        logger.info(f"\n✅ All plots saved to: {self.output_dir}")


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate plots from COCO data')
    parser.add_argument('--coco_dir', type=str, required=True,
                        help='COCO format directory')
    parser.add_argument('--output_dir', type=str, default='plots',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Construct coco_paths
    coco_dir = Path(args.coco_dir)
    coco_paths = {
        'train_dir': str(coco_dir / 'train' / 'images'),
        'val_dir': str(coco_dir / 'val' / 'images'),
        'test_dir': str(coco_dir / 'test' / 'images'),
        'train_ann': str(coco_dir / 'annotations' / 'instances_train.json'),
        'val_ann': str(coco_dir / 'annotations' / 'instances_val.json'),
        'test_ann': str(coco_dir / 'annotations' / 'instances_test.json')
    }
    
    # Generate plots
    plotter = PlotGenerator(coco_paths, output_dir=args.output_dir)
    plotter.generate_all_plots()


if __name__ == "__main__":
    main()

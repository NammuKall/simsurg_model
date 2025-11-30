#data.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Data Pipeline Orchestrator
Runs the complete data processing pipeline:
1. Extract frames from videos (parallel)
2. Wrangle and validate data
3. Create COCO format JSON
4. Generate plots and visualizations
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import time

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data.data_extractor import VideoFrameExtractor
from src.data.data_wrangler import DataWrangler
from src.data.coco_json import COCOJSONGenerator
from src.data.plots import PlotGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """Complete data processing pipeline"""
    
    def __init__(self, data_dir, output_dir='data/coco_format', 
                 plots_dir='plots', fps=1, num_workers=None):
        """
        Initialize the data pipeline
        
        Args:
            data_dir (str): Base directory containing SimSurgSkill dataset
            output_dir (str): Output directory for COCO format
            plots_dir (str): Output directory for plots
            fps (int): Frames per second to extract
            num_workers (int): Number of parallel workers
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.plots_dir = Path(plots_dir)
        self.fps = fps
        self.num_workers = num_workers
        
        # Verify data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        logger.info("="*60)
        logger.info("DATA PIPELINE INITIALIZED")
        logger.info("="*60)
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Plots directory: {self.plots_dir}")
        logger.info(f"FPS: {self.fps}")
        logger.info(f"Workers: {self.num_workers if self.num_workers else 'auto'}")
    
    def step1_extract_frames(self, skip_if_exists=True):
        """
        Step 1: Extract frames from videos
        
        Args:
            skip_if_exists (bool): Skip extraction if images already exist
            
        Returns:
            dict: Extraction results
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 1: EXTRACTING FRAMES FROM VIDEOS")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Check if extraction needed
        if skip_if_exists:
            splits_exist = all([
                (self.data_dir / split / 'videos' / 'fps1').exists() and
                list((self.data_dir / split / 'videos' / 'fps1').glob('*.jpeg'))
                for split in ['train_v1', 'train_v2', 'test']
                if (self.data_dir / split / 'videos' / 'fps1').exists()
            ])
            
            if splits_exist:
                logger.info("âœ“ Frames already extracted. Skipping extraction.")
                logger.info("  Use --force-extract to re-extract frames")
                return {'skipped': True}
        
        # Create extractor
        extractor = VideoFrameExtractor(fps=self.fps, num_workers=self.num_workers)
        
        # Extract frames from all splits
        results = extractor.extract_all_splits(self.data_dir)
        
        elapsed = time.time() - start_time
        logger.info(f"\nâœ… Frame extraction complete in {elapsed:.2f} seconds")
        
        return results
    
    def step2_wrangle_data(self):
        """
        Step 2: Wrangle and validate data
        
        Returns:
            dict: Wrangled data
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 2: WRANGLING AND VALIDATING DATA")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Create wrangler
        wrangler = DataWrangler(self.data_dir)
        
        # Wrangle all splits
        wrangled_data = wrangler.wrangle_all_splits()
        
        # Save wrangled data
        wrangled_file = 'wrangled_data.json'
        wrangler.save_wrangled_data(wrangled_file)
        
        elapsed = time.time() - start_time
        logger.info(f"\nâœ… Data wrangling complete in {elapsed:.2f} seconds")
        
        return wrangled_data
    
    def step3_create_coco_json(self, wrangled_data, train_ratio=0.7, val_ratio=0.15):
        """
        Step 3: Create COCO format JSON
        
        Args:
            wrangled_data (dict): Wrangled data from step 2
            train_ratio (float): Training set ratio
            val_ratio (float): Validation set ratio
            
        Returns:
            dict: COCO paths
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 3: CREATING COCO FORMAT JSON")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Create generator
        generator = COCOJSONGenerator(output_dir=self.output_dir)
        
        # Generate COCO format
        coco_paths = generator.generate_from_wrangled_data(
            wrangled_data,
            train_ratio=train_ratio,
            val_ratio=val_ratio
        )
        
        elapsed = time.time() - start_time
        logger.info(f"\nâœ… COCO JSON creation complete in {elapsed:.2f} seconds")
        
        return coco_paths
    
    def step4_generate_plots(self, coco_paths):
        """
        Step 4: Generate plots and visualizations
        
        Args:
            coco_paths (dict): Paths to COCO datasets
            
        Returns:
            None
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 4: GENERATING PLOTS AND VISUALIZATIONS")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Create plot generator
        plotter = PlotGenerator(coco_paths, output_dir=self.plots_dir)
        
        # Generate all plots
        plotter.generate_all_plots()
        
        elapsed = time.time() - start_time
        logger.info(f"\nâœ… Plot generation complete in {elapsed:.2f} seconds")
    
    def run_full_pipeline(self, skip_extraction=True, train_ratio=0.7, val_ratio=0.15):
        """
        Run the complete data processing pipeline
        
        Args:
            skip_extraction (bool): Skip frame extraction if images exist
            train_ratio (float): Training set ratio
            val_ratio (float): Validation set ratio
            
        Returns:
            dict: COCO paths
        """
        total_start = time.time()
        
        logger.info("\n" + "ðŸš€ "*20)
        logger.info("STARTING FULL DATA PIPELINE")
        logger.info("ðŸš€ "*20 + "\n")
        
        try:
            # Step 1: Extract frames
            extraction_results = self.step1_extract_frames(skip_if_exists=skip_extraction)
            
            # Step 2: Wrangle data
            wrangled_data = self.step2_wrangle_data()
            
            # Step 3: Create COCO JSON
            coco_paths = self.step3_create_coco_json(wrangled_data, train_ratio, val_ratio)
            
            # Step 4: Generate plots
            self.step4_generate_plots(coco_paths)
            
            # Pipeline complete
            total_elapsed = time.time() - total_start
            
            logger.info("\n" + "ðŸŽ‰ "*20)
            logger.info("PIPELINE COMPLETE!")
            logger.info("ðŸŽ‰ "*20)
            logger.info(f"\nTotal time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
            logger.info(f"\nOutputs:")
            logger.info(f"  â”œâ”€ COCO format: {self.output_dir}")
            logger.info(f"  â”‚  â”œâ”€ train/images/")
            logger.info(f"  â”‚  â”œâ”€ val/images/")
            logger.info(f"  â”‚  â”œâ”€ test/images/")
            logger.info(f"  â”‚  â””â”€ annotations/")
            logger.info(f"  â”‚     â”œâ”€ instances_train.json")
            logger.info(f"  â”‚     â”œâ”€ instances_val.json")
            logger.info(f"  â”‚     â””â”€ instances_test.json")
            logger.info(f"  â””â”€ Plots: {self.plots_dir}")
            logger.info(f"     â”œâ”€ dataset_overview.png")
            logger.info(f"     â”œâ”€ class_distribution_by_split.png")
            logger.info(f"     â”œâ”€ bbox_statistics.png")
            logger.info(f"     â””â”€ sample_annotations.png")
            
            logger.info(f"\nðŸ“Š Next steps:")
            logger.info(f"  1. Review plots in {self.plots_dir}")
            logger.info(f"  2. Verify COCO format in {self.output_dir}")
            logger.info(f"  3. Start training with: python main.py")
            
            return coco_paths
            
        except Exception as e:
            logger.error(f"\nâŒ Pipeline failed: {e}")
            raise


def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(
        description='SimSurgSkill Data Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python data.py --data_dir data/simsurgskill_2021_dataset
  
  # Run with custom split ratios
  python data.py --data_dir data/simsurgskill_2021_dataset --train_ratio 0.8 --val_ratio 0.1
  
  # Force re-extraction of frames
  python data.py --data_dir data/simsurgskill_2021_dataset --force-extract
  
  # Use specific number of workers
  python data.py --data_dir data/simsurgskill_2021_dataset --workers 8
        """
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Base directory containing SimSurgSkill dataset'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/coco_format',
        help='Output directory for COCO format (default: data/coco_format)'
    )
    
    parser.add_argument(
        '--plots_dir',
        type=str,
        default='plots',
        help='Output directory for plots (default: plots)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=1,
        help='Frames per second to extract (default: 1)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: all CPUs)'
    )
    
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )
    
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    
    parser.add_argument(
        '--force-extract',
        action='store_true',
        help='Force re-extraction of frames even if they exist'
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0 or test_ratio > 1:
        parser.error("train_ratio + val_ratio must be <= 1.0")
    
    logger.info(f"Test ratio (calculated): {test_ratio:.2f}")
    
    # Create pipeline
    pipeline = DataPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        plots_dir=args.plots_dir,
        fps=args.fps,
        num_workers=args.workers
    )
    
    # Run pipeline
    coco_paths = pipeline.run_full_pipeline(
        skip_extraction=not args.force_extract,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    return coco_paths


if __name__ == "__main__":
    main()

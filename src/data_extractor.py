#data_extractor.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Extractor - Extract frames from videos in parallel
Uses all available CPU cores for maximum speed
"""

import os
import cv2
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoFrameExtractor:
    """Extract frames from surgical videos using parallel processing"""
    
    def __init__(self, fps=1, num_workers=None):
        """
        Initialize the extractor
        
        Args:
            fps (int): Frames per second to extract (default: 1)
            num_workers (int): Number of parallel workers (default: all CPUs)
        """
        self.fps = fps
        self.num_workers = num_workers if num_workers else cpu_count()
        logger.info(f"Initialized VideoFrameExtractor with {self.num_workers} workers, fps={fps}")
    
    @staticmethod
    def extract_frame_from_video(video_path, output_dir, fps=1):
        """
        Extract a single frame from a video file
        
        Args:
            video_path (str): Path to video file
            output_dir (str): Directory to save extracted frame
            fps (int): Frame rate (1 = first frame only)
            
        Returns:
            dict: Status with video_path, success, and message
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Open video file
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                return {
                    'video_path': str(video_path),
                    'success': False,
                    'message': 'Failed to open video'
                }
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame interval
            if fps > 0 and video_fps > 0:
                frame_interval = int(video_fps / fps)
            else:
                frame_interval = 1
            
            # For fps=1, just extract the first frame
            frame_count = 0
            extracted_count = 0
            
            while frame_count < total_frames:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Extract frame at specified interval
                if frame_count % frame_interval == 0 or fps == 1:
                    # Generate output filename
                    video_name = Path(video_path).stem
                    if fps == 1 and extracted_count == 0:
                        # For single frame extraction, use simple naming
                        output_filename = f"{video_name}.jpeg"
                    else:
                        output_filename = f"{video_name}_frame{extracted_count:04d}.jpeg"
                    
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Save frame
                    cv2.imwrite(output_path, frame)
                    extracted_count += 1
                    
                    # If fps=1, only extract first frame
                    if fps == 1:
                        break
                
                frame_count += 1
            
            cap.release()
            
            return {
                'video_path': str(video_path),
                'success': True,
                'message': f'Extracted {extracted_count} frame(s)',
                'frames_extracted': extracted_count
            }
            
        except Exception as e:
            return {
                'video_path': str(video_path),
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
    def extract_frames_from_directory(self, video_dir, output_dir=None):
        """
        Extract frames from all videos in a directory using parallel processing
        
        Args:
            video_dir (str): Directory containing video files
            output_dir (str): Directory to save extracted frames (default: same as video_dir)
            
        Returns:
            list: Results from all video extractions
        """
        video_dir = Path(video_dir)
        
        # Use same directory if output not specified
        if output_dir is None:
            output_dir = video_dir
        else:
            output_dir = Path(output_dir)
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(list(video_dir.glob(f'*{ext}')))
        
        if not video_files:
            logger.warning(f"No video files found in {video_dir}")
            return []
        
        logger.info(f"Found {len(video_files)} video files in {video_dir}")
        
        # Prepare arguments for parallel processing
        args_list = [(str(vf), str(output_dir), self.fps) for vf in video_files]
        
        # Extract frames in parallel
        results = []
        with Pool(processes=self.num_workers) as pool:
            # Use starmap to pass multiple arguments
            with tqdm(total=len(args_list), desc=f"Extracting frames from {video_dir.name}") as pbar:
                for result in pool.starmap(self.extract_frame_from_video, args_list):
                    results.append(result)
                    pbar.update(1)
        
        # Log summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        logger.info(f"Extraction complete: {successful} successful, {failed} failed")
        
        return results
    
    def extract_all_splits(self, base_dir, splits=['train_v1', 'train_v2', 'test']):
        """
        Extract frames from all dataset splits
        
        Args:
            base_dir (str): Base directory containing the dataset
            splits (list): List of split names to process
            
        Returns:
            dict: Results for each split
        """
        base_dir = Path(base_dir)
        all_results = {}
        
        for split in splits:
            video_dir = base_dir / split / 'videos' / f'fps{self.fps}'
            
            if not video_dir.exists():
                logger.warning(f"Directory not found: {video_dir}")
                continue
            
            logger.info(f"\nProcessing {split} split...")
            results = self.extract_frames_from_directory(video_dir)
            all_results[split] = results
        
        # Print overall summary
        logger.info("\n" + "="*60)
        logger.info("EXTRACTION SUMMARY")
        logger.info("="*60)
        
        for split, results in all_results.items():
            successful = sum(1 for r in results if r['success'])
            total_frames = sum(r.get('frames_extracted', 0) for r in results if r['success'])
            logger.info(f"{split}: {successful}/{len(results)} videos processed, {total_frames} frames extracted")
        
        return all_results


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract frames from surgical videos')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory containing the dataset')
    parser.add_argument('--fps', type=int, default=1,
                        help='Frames per second to extract (default: 1)')
    parser.add_argument('--splits', nargs='+', default=['train_v1', 'train_v2', 'test'],
                        help='Dataset splits to process')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: all CPUs)')
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = VideoFrameExtractor(fps=args.fps, num_workers=args.workers)
    
    # Extract frames from all splits
    results = extractor.extract_all_splits(args.data_dir, splits=args.splits)
    
    logger.info("\nExtraction complete!")
    
    return results


if __name__ == "__main__":
    main()

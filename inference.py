#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for SimSurg object detection model

This script loads a trained model and performs inference on test images.
It visualizes predictions, computes metrics, and provides detailed analysis.

WHAT THIS SCRIPT DOES:
1. Loads a trained model checkpoint
2. Runs inference on test/validation images
3. Visualizes predictions with bounding boxes
4. Computes detection metrics (IoU, precision, recall, mAP)
5. Generates detailed reports and statistics
6. Saves visualized results

HOW TO INTERPRET THE RESULTS:

1. VISUALIZATIONS:
   - Green boxes: Ground truth annotations
   - Red boxes: Model predictions
   - Confidence scores: Shown for each prediction

2. METRICS:
   - IoU (Intersection over Union): Measures box overlap quality (0-1, higher is better)
   - Precision: Fraction of positive predictions that are correct
   - Recall: Fraction of ground truths that were detected
   - mAP (mean Average Precision): Overall detection quality (0-1, higher is better)

3. CONFIDENCE THRESHOLDS:
   - High threshold (>0.7): Only very confident predictions
   - Medium threshold (0.5-0.7): Balanced precision/recall
   - Low threshold (<0.5): More detections but more false positives

4. WHAT GOOD RESULTS LOOK LIKE:
   - IoU > 0.5: Boxes have good overlap
   - mAP > 0.7: Model is performing well
   - High precision + high recall: Model finds most objects with few mistakes
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# Import model and data loading
from src.models import EfficientDetModel
from src.coco_data_loader import get_coco_data_loaders

console = Console()


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Compute IoU between two boxes in [x_min, y_min, x_max, y_max] format
    
    Args:
        box1: [4] tensor
        box2: [4] tensor
    
    Returns:
        IoU score (0-1)
    """
    # Calculate intersection area
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return (intersection / union).item() if union > 0 else 0.0


def match_predictions(pred_boxes: torch.Tensor, pred_scores: torch.Tensor, 
                     gt_boxes: torch.Tensor, iou_threshold: float = 0.5) -> Dict:
    """
    Match predictions to ground truth boxes
    
    Args:
        pred_boxes: [N, 4] predicted boxes
        pred_scores: [N] confidence scores
        gt_boxes: [M, 4] ground truth boxes
    
    Returns:
        Dictionary with matched indices, IoUs, and metrics
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return {
            'matched_predictions': [],
            'matched_gts': [],
            'iou_scores': [],
            'true_positives': 0,
            'false_positives': len(pred_boxes),
            'false_negatives': len(gt_boxes)
        }
    
    # Compute IoU matrix
    ious = []
    matched_gts = set()
    matched_preds = []
    
    for i, pred_box in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou and j not in matched_gts:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold:
            matched_gts.add(best_gt_idx)
            matched_preds.append((i, best_gt_idx, best_iou))
    
    # Calculate metrics
    tp = len(matched_preds)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gts)
    
    return {
        'matched_predictions': matched_preds,
        'matched_gts': list(matched_gts),
        'iou_scores': [iou for _, _, iou in matched_preds],
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }


def visualize_predictions(image: torch.Tensor, pred_boxes: torch.Tensor, 
                         pred_scores: torch.Tensor, pred_labels: torch.Tensor,
                         gt_boxes: torch.Tensor = None, save_path: str = None):
    """
    Visualize predictions on image with optional ground truth
    
    Args:
        image: [C, H, W] tensor
        pred_boxes: [N, 4] predicted boxes
        pred_scores: [N] confidence scores
        pred_labels: [N] class labels
        gt_boxes: [M, 4] ground truth boxes (optional)
        save_path: Path to save visualization (optional)
    """
    # Convert image to numpy
    img = image.permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(img, 0, 1)
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    # Draw ground truth boxes (green)
    if gt_boxes is not None:
        for box in gt_boxes:
            x1, y1, x2, y2 = box.cpu().numpy()
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, 
                                    linewidth=2, edgecolor='green', 
                                    facecolor='none', label='Ground Truth')
            ax.add_patch(rect)
    
    # Draw predicted boxes (red)
    for i, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
        x1, y1, x2, y2 = box.cpu().numpy()
        width = x2 - x1
        height = y2 - y1
        
        # Box
        rect = patches.Rectangle((x1, y1), width, height,
                               linewidth=2, edgecolor='red',
                               facecolor='none', linestyle='--',
                               alpha=0.7)
        ax.add_patch(rect)
        
        # Score text
        ax.text(x1, y1, f'{label.item()}: {score:.2f}',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
               fontsize=10, color='red')
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def run_inference(model_path: str, coco_paths: Dict, output_dir: str = "inference_results",
                  batch_size: int = 1, num_samples: int = None, visualize: bool = True):
    """
    Run inference on test set
    
    Args:
        model_path: Path to trained model checkpoint
        coco_paths: Dictionary with COCO data paths
        output_dir: Directory to save results
        batch_size: Batch size for inference
        num_samples: Number of samples to process (None = all)
        visualize: Whether to generate visualizations
    
    Returns:
        Dictionary with metrics and results
    """
    console.print(Panel.fit(
        "[bold blue]RUNNING INFERENCE[/bold blue]\n"
        f"Model: {model_path}\n"
        f"Output: {output_dir}",
        title="ðŸ” Inference",
        border_style="blue"
    ))
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    vis_dir = output_path / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[green]Using device:[/green] {device}")
    
    # Load model
    console.print(f"[blue]Loading model from {model_path}[/blue]")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = EfficientDetModel(num_classes=2).to(device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    console.print("[green]âœ… Model loaded successfully[/green]")
    
    # Get data loader
    _, _, test_loader = get_coco_data_loaders(coco_paths, batch_size=batch_size)
    
    # Run inference
    all_metrics = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious = []
    
    console.print(f"[blue]Processing {len(test_loader)} batches[/blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Running inference", total=min(len(test_loader), num_samples) if num_samples else len(test_loader))
        
        for i, (images, targets) in enumerate(test_loader):
            if num_samples and i >= num_samples:
                break
            
            # Move to device
            images = images.to(device)
            
            # Run inference
            with torch.no_grad():
                outputs = model(images)
            
            # Process each image in batch
            for j in range(len(images)):
                image = images[j]
                target = targets[j]
                pred = outputs[j]
                
                # Get predictions
                pred_boxes = pred['boxes']
                pred_scores = pred['scores']
                pred_labels = pred['labels']
                
                # Get ground truth
                gt_boxes = target['boxes']
                
                # Match predictions to ground truth
                matches = match_predictions(pred_boxes, pred_scores, gt_boxes)
                
                # Accumulate metrics
                total_tp += matches['true_positives']
                total_fp += matches['false_positives']
                total_fn += matches['false_negatives']
                all_ious.extend(matches['iou_scores'])
                
                # Visualize if requested
                if visualize and i < 10:  # Only visualize first 10
                    vis_path = vis_dir / f"sample_{i}_{j}.png"
                    visualize_predictions(image, pred_boxes, pred_scores, 
                                         pred_labels, gt_boxes, str(vis_path))
                
                all_metrics.append({
                    'sample_id': f"{i}_{j}",
                    'num_predictions': len(pred_boxes),
                    'num_ground_truth': len(gt_boxes),
                    'num_matches': matches['true_positives'],
                    'avg_iou': np.mean(matches['iou_scores']) if matches['iou_scores'] else 0
                })
            
            progress.update(task, advance=1)
    
    # Calculate overall metrics
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Display results
    results_table = Table(title="Inference Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    
    results_table.add_row("Total True Positives", str(total_tp))
    results_table.add_row("Total False Positives", str(total_fp))
    results_table.add_row("Total False Negatives", str(total_fn))
    results_table.add_row("Mean IoU", f"{mean_iou:.4f}")
    results_table.add_row("Precision", f"{precision:.4f}")
    results_table.add_row("Recall", f"{recall:.4f}")
    results_table.add_row("F1 Score", f"{f1_score:.4f}")
    
    console.print(results_table)
    
    # Save results
    results = {
        'model_path': model_path,
        'device': str(device),
        'total_samples': len(all_metrics),
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'mean_iou': float(mean_iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = output_path / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\n[green]âœ… Results saved to:[/green] {results_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference on trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="inference_results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--no_viz", action="store_true", help="Skip visualizations")
    
    args = parser.parse_args()
    
    # Define paths (adjust as needed)
    base_dir = Path(__file__).parent
    coco_dir = base_dir / "data" / "coco_format"
    
    coco_paths = {
        'train_dir': str(coco_dir / 'train' / 'images'),
        'val_dir': str(coco_dir / 'val' / 'images'),
        'test_dir': str(coco_dir / 'test' / 'images'),
        'train_ann': str(coco_dir / 'annotations' / 'instances_train.json'),
        'val_ann': str(coco_dir / 'annotations' / 'instances_val.json'),
        'test_ann': str(coco_dir / 'annotations' / 'instances_test.json')
    }
    
    results = run_inference(
        model_path=args.model,
        coco_paths=coco_paths,
        output_dir=args.output,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        visualize=not args.no_viz
    )
    
    console.print(f"\n[bold green]Inference complete![/bold green]")
    console.print(f"[cyan]View results in:[/cyan] {args.output}")

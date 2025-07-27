import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from sklearn.metrics import average_precision_score
import seaborn as sns

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1, box2: [x1, y1, x2, y2] format
    
    Returns:
        float: IoU value
    """
    # Get intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def convert_bbox_format(bbox, format_from='coco', format_to='xyxy'):
    """
    Convert bounding box format
    
    Args:
        bbox: bounding box coordinates
        format_from: 'coco' [x, y, w, h] or 'xyxy' [x1, y1, x2, y2]
        format_to: 'coco' [x, y, w, h] or 'xyxy' [x1, y1, x2, y2]
    """
    if format_from == 'coco' and format_to == 'xyxy':
        return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    elif format_from == 'xyxy' and format_to == 'coco':
        return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
    else:
        return bbox

def evaluate_model(model, data_loader, device, confidence_thresholds=None, iou_threshold=0.5):
    """
    Evaluate object detection model with various metrics
    
    Args:
        model: trained model
        data_loader: test data loader
        device: torch device
        confidence_thresholds: list of confidence thresholds to evaluate
        iou_threshold: IoU threshold for considering a detection as correct
    
    Returns:
        dict: evaluation results
    """
    if confidence_thresholds is None:
        confidence_thresholds = np.arange(0.1, 1.0, 0.1)
    
    model.eval()
    
    # Store all predictions and ground truths
    all_predictions = []
    all_ground_truths = []
    
    print("Collecting predictions and ground truths...")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            images = images.to(device)
            
            # Get model predictions
            predictions = model(images)
            
            # Process each image in the batch
            for i in range(len(images)):
                pred = predictions[i] if isinstance(predictions, list) else {
                    'boxes': predictions['boxes'][i],
                    'scores': predictions['scores'][i],
                    'labels': predictions['labels'][i]
                }
                
                # Store predictions
                pred_data = {
                    'boxes': pred['boxes'].cpu().numpy(),
                    'scores': pred['scores'].cpu().numpy(),
                    'labels': pred['labels'].cpu().numpy(),
                    'image_id': batch_idx * len(images) + i
                }
                all_predictions.append(pred_data)
                
                # Store ground truths
                gt_data = {
                    'boxes': targets[i]['boxes'].cpu().numpy(),
                    'labels': targets[i]['labels'].cpu().numpy(),
                    'image_id': batch_idx * len(images) + i
                }
                all_ground_truths.append(gt_data)
            
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(data_loader)} batches")
    
    print("Computing metrics...")
    
    # Calculate metrics for each confidence threshold
    results = {
        'confidence_thresholds': confidence_thresholds,
        'precision': [],
        'recall': [],
        'miou': [],
        'per_class_metrics': defaultdict(list)
    }
    
    for conf_thresh in confidence_thresholds:
        precision, recall, miou, per_class = calculate_metrics_at_threshold(
            all_predictions, all_ground_truths, conf_thresh, iou_threshold
        )
        
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['miou'].append(miou)
        
        for class_id, metrics in per_class.items():
            results['per_class_metrics'][class_id].append(metrics)
    
    # Calculate Average Precision (AP) for P-R curve
    results['average_precision'] = calculate_average_precision(all_predictions, all_ground_truths, iou_threshold)
    
    return results

def calculate_metrics_at_threshold(predictions, ground_truths, conf_threshold, iou_threshold):
    """
    Calculate precision, recall, and mIoU at a specific confidence threshold
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_iou = 0
    matched_predictions = 0
    
    per_class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'iou_sum': 0, 'matches': 0})
    
    # Group by image
    pred_by_image = defaultdict(list)
    gt_by_image = defaultdict(list)
    
    for pred in predictions:
        pred_by_image[pred['image_id']].append(pred)
    
    for gt in ground_truths:
        gt_by_image[gt['image_id']].append(gt)
    
    # Process each image
    for image_id in gt_by_image.keys():
        gt_list = gt_by_image[image_id]
        pred_list = pred_by_image.get(image_id, [])
        
        # Filter predictions by confidence
        filtered_preds = []
        for pred in pred_list:
            valid_indices = pred['scores'] >= conf_threshold
            if np.any(valid_indices):
                filtered_preds.append({
                    'boxes': pred['boxes'][valid_indices],
                    'scores': pred['scores'][valid_indices],
                    'labels': pred['labels'][valid_indices]
                })
        
        # Match predictions to ground truths
        for gt in gt_list:
            gt_boxes = gt['boxes']
            gt_labels = gt['labels']
            matched_gt = set()
            
            for pred in filtered_preds:
                pred_boxes = pred['boxes']
                pred_labels = pred['labels']
                matched_pred = set()
                
                # For each ground truth box
                for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    if gt_idx in matched_gt:
                        continue
                    
                    best_iou = 0
                    best_pred_idx = -1
                    
                    # Find best matching prediction
                    for pred_idx, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                        if pred_idx in matched_pred or pred_label != gt_label:
                            continue
                        
                        # Convert boxes to xyxy format if needed
                        gt_box_xyxy = convert_bbox_format(gt_box, 'coco', 'xyxy')
                        pred_box_xyxy = convert_bbox_format(pred_box, 'coco', 'xyxy')
                        
                        iou = calculate_iou(gt_box_xyxy, pred_box_xyxy)
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_pred_idx = pred_idx
                    
                    # Check if match is good enough
                    if best_iou >= iou_threshold and best_pred_idx != -1:
                        true_positives += 1
                        matched_gt.add(gt_idx)
                        matched_pred.add(best_pred_idx)
                        total_iou += best_iou
                        matched_predictions += 1
                        
                        # Per-class metrics
                        per_class_metrics[gt_label]['tp'] += 1
                        per_class_metrics[gt_label]['iou_sum'] += best_iou
                        per_class_metrics[gt_label]['matches'] += 1
                
                # Count false positives (unmatched predictions)
                for pred_idx, pred_label in enumerate(pred_labels):
                    if pred_idx not in matched_pred:
                        false_positives += 1
                        per_class_metrics[pred_label]['fp'] += 1
            
            # Count false negatives (unmatched ground truths)
            for gt_idx, gt_label in enumerate(gt_labels):
                if gt_idx not in matched_gt:
                    false_negatives += 1
                    per_class_metrics[gt_label]['fn'] += 1
    
    # Calculate overall metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    miou = total_iou / matched_predictions if matched_predictions > 0 else 0
    
    # Calculate per-class metrics
    per_class_results = {}
    for class_id, metrics in per_class_metrics.items():
        class_precision = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0
        class_recall = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0
        class_miou = metrics['iou_sum'] / metrics['matches'] if metrics['matches'] > 0 else 0
        
        per_class_results[class_id] = {
            'precision': class_precision,
            'recall': class_recall,
            'miou': class_miou
        }
    
    return precision, recall, miou, per_class_results

def calculate_average_precision(predictions, ground_truths, iou_threshold):
    """
    Calculate Average Precision for P-R curve
    """
    # Collect all prediction scores and their corresponding precision/recall
    all_scores = []
    all_labels = []
    
    # This is a simplified version - you might want to implement per-class AP
    for pred in predictions:
        all_scores.extend(pred['scores'])
        # For simplicity, treating all detections as positive class
        all_labels.extend([1] * len(pred['scores']))
    
    if len(all_scores) == 0:
        return 0.0
    
    # Sort by confidence score
    sorted_indices = np.argsort(all_scores)[::-1]
    sorted_scores = np.array(all_scores)[sorted_indices]
    
    # Calculate precision and recall at each score threshold
    precisions = []
    recalls = []
    
    for i, score in enumerate(sorted_scores):
        # Use the score as threshold and calculate metrics
        precision, recall, _, _ = calculate_metrics_at_threshold(
            predictions, ground_truths, score, iou_threshold
        )
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate AP using sklearn
    if len(set(recalls)) > 1:  # Need variation in recall values
        ap = average_precision_score([1] * len(recalls), recalls)
    else:
        ap = 0.0
    
    return ap

def plot_evaluation_results(results, save_path=None):
    """
    Plot evaluation results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Precision vs Confidence Threshold
    axes[0, 0].plot(results['confidence_thresholds'], results['precision'], 'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Confidence Threshold')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision vs Confidence Threshold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # Plot 2: Recall vs Confidence Threshold
    axes[0, 1].plot(results['confidence_thresholds'], results['recall'], 'r-o', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Confidence Threshold')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_title('Recall vs Confidence Threshold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Plot 3: mIoU vs Confidence Threshold
    axes[1, 0].plot(results['confidence_thresholds'], results['miou'], 'g-o', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Confidence Threshold')
    axes[1, 0].set_ylabel('mIoU')
    axes[1, 0].set_title('mIoU vs Confidence Threshold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Plot 4: Precision-Recall Curve
    axes[1, 1].plot(results['recall'], results['precision'], 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].set_title(f'Precision-Recall Curve (AP={results["average_precision"]:.3f})')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation plots saved to {save_path}")
    
    plt.show()

def print_evaluation_summary(results):
    """
    Print summary of evaluation results
    """
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    # Find best metrics
    best_precision_idx = np.argmax(results['precision'])
    best_recall_idx = np.argmax(results['recall'])
    best_miou_idx = np.argmax(results['miou'])
    
    print(f"Best Precision: {results['precision'][best_precision_idx]:.3f} at threshold {results['confidence_thresholds'][best_precision_idx]:.2f}")
    print(f"Best Recall: {results['recall'][best_recall_idx]:.3f} at threshold {results['confidence_thresholds'][best_recall_idx]:.2f}")
    print(f"Best mIoU: {results['miou'][best_miou_idx]:.3f} at threshold {results['confidence_thresholds'][best_miou_idx]:.2f}")
    print(f"Average Precision (AP): {results['average_precision']:.3f}")
    
    # F1 scores
    f1_scores = []
    for p, r in zip(results['precision'], results['recall']):
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1_scores.append(f1)
    
    best_f1_idx = np.argmax(f1_scores)
    print(f"Best F1 Score: {f1_scores[best_f1_idx]:.3f} at threshold {results['confidence_thresholds'][best_f1_idx]:.2f}")
    
    print("="*50)

# Integration function to add to your main script
def run_evaluation(model, test_loader, device, save_dir=None):
    """
    Run complete evaluation pipeline
    """
    print("Starting evaluation...")
    
    # Define confidence thresholds
    confidence_thresholds = np.arange(0.1, 1.0, 0.05)
    
    # Run evaluation
    results = evaluate_model(
        model, 
        test_loader, 
        device, 
        confidence_thresholds=confidence_thresholds,
        iou_threshold=0.5
    )
    
    # Print summary
    print_evaluation_summary(results)
    
    # Plot results
    if save_dir:
        plot_save_path = os.path.join(save_dir, 'evaluation_plots.png')
        os.makedirs(save_dir, exist_ok=True)
    else:
        plot_save_path = None
    
    plot_evaluation_results(results, plot_save_path)
    
    # Save results to JSON
    if save_dir:
        results_save_path = os.path.join(save_dir, 'evaluation_results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = {str(k): v for k, v in value.items()}
            else:
                json_results[key] = value
        
        with open(results_save_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to {results_save_path}")
    
    return results

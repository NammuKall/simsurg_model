#training.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training functions for model training and validation
"""
import time
import logging
import numpy as np
import torch
import wandb
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from src.testing import compute_detection_metrics

# Get logger and console from main module context
logger = logging.getLogger()
console = Console()


def train_epoch(model, train_loader, optimizer, device, epoch, num_epochs, wandb_api_key, batch_size, val_loader):
    """Train for one epoch"""
    model.train()
    train_loss = 0.0
    num_batches = 0
    successful_batches = 0
    failed_batches = 0
    batch_losses = []
    batch_times = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        train_task = progress.add_task(f"Training Epoch {epoch+1}", total=len(train_loader))
        
        for i, (images, targets) in enumerate(train_loader):
            batch_start_time = time.time()
            try:
                images = images.to(device)
                
                for j in range(len(targets)):
                    targets[j]['boxes'] = targets[j]['boxes'].to(device)
                    targets[j]['labels'] = targets[j]['labels'].to(device)
                    targets[j]['image_id'] = targets[j]['image_id'].to(device)
                
                outputs = model(images, targets)
                
                if isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"Invalid loss value at batch {i}: {loss.item()}")
                        failed_batches += 1
                        continue
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    grad_norm = 0.0
                    for param in model.parameters():
                        if param.grad is not None:
                            grad_norm += param.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5
                    
                    if torch.isnan(torch.tensor(grad_norm)) or torch.isinf(torch.tensor(grad_norm)):
                        logger.warning(f"Invalid gradient norm at batch {i}: {grad_norm}")
                        failed_batches += 1
                        continue
                    
                    optimizer.step()
                    
                    loss_value = loss.item()
                    train_loss += loss_value
                    batch_losses.append(loss_value)
                    num_batches += 1
                    successful_batches += 1
                    
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
                    samples_per_sec = batch_size / batch_time if batch_time > 0 else 0
                    
                    if wandb_api_key and i % 10 == 0:
                        log_dict = {
                            "batch_train_loss": loss.item(),
                            "batch": i + epoch * len(train_loader),
                            "grad_norm": grad_norm,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            "batch_time": batch_time,
                            "samples_per_sec": samples_per_sec
                        }
                        
                        if torch.cuda.is_available():
                            log_dict["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(0) / 1024**2
                            log_dict["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(0) / 1024**2
                        
                        wandb.log(log_dict)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f'Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, '
                                   f'Grad Norm: {grad_norm:.4f}, Batch Time: {batch_time:.3f}s, '
                                   f'Speed: {samples_per_sec:.1f} samples/sec')
                    
                    if (i + 1) % 100 == 0 and wandb_api_key:
                        logger.info(f"Running periodic evaluation at batch {i+1}")
                        try:
                            periodic_metrics = compute_detection_metrics(model, val_loader, device, num_samples=50)
                            
                            wandb.log({
                                "periodic_val_mean_iou": periodic_metrics['mean_iou'],
                                "periodic_val_precision": periodic_metrics['precision'],
                                "periodic_val_recall": periodic_metrics['recall'],
                                "periodic_val_f1_score": periodic_metrics['f1_score'],
                                "periodic_batch": i + 1 + epoch * len(train_loader)
                            })
                            
                            logger.info(f"Periodic Val Eval (50 samples) - IoU: {periodic_metrics['mean_iou']:.3f}, "
                                       f"Precision: {periodic_metrics['precision']:.3f}, "
                                       f"Recall: {periodic_metrics['recall']:.3f}, "
                                       f"F1: {periodic_metrics['f1_score']:.3f} "
                                       f"(TP: {periodic_metrics['true_positives']}, "
                                       f"FP: {periodic_metrics['false_positives']}, "
                                       f"FN: {periodic_metrics['false_negatives']})")
                        except Exception as e:
                            logger.warning(f"Periodic evaluation failed: {e}")
                else:
                    logger.warning(f"Unexpected output format at batch {i}: {type(outputs)}")
                    failed_batches += 1
                    continue
                    
            except Exception as e:
                logger.error(f"Error in training batch {i}: {e}")
                logger.debug(f"Batch {i} error details:", exc_info=True)
                failed_batches += 1
                continue
            
            progress.update(train_task, advance=1)
    
    avg_train_loss = train_loss / num_batches if num_batches > 0 else 0
    avg_batch_time = np.mean(batch_times) if batch_times else 0
    avg_samples_per_sec = np.mean([batch_size / t for t in batch_times if t > 0]) if batch_times else 0
    
    return {
        'avg_loss': avg_train_loss,
        'batch_losses': batch_losses,
        'successful_batches': successful_batches,
        'failed_batches': failed_batches,
        'batch_times': batch_times,
        'avg_batch_time': avg_batch_time,
        'avg_samples_per_sec': avg_samples_per_sec
    }


def validate_epoch(model, val_loader, device, epoch, num_epochs):
    """Validate for one epoch"""
    model.eval()
    val_loss = 0.0
    num_val_batches = 0
    val_successful_batches = 0
    val_failed_batches = 0
    val_batch_losses = []
    
    with torch.no_grad():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            val_task = progress.add_task(f"Validation Epoch {epoch+1}", total=len(val_loader))
            
            for i, (images, targets) in enumerate(val_loader):
                try:
                    images = images.to(device)
                    
                    for j in range(len(targets)):
                        targets[j]['boxes'] = targets[j]['boxes'].to(device)
                        targets[j]['labels'] = targets[j]['labels'].to(device)
                        targets[j]['image_id'] = targets[j]['image_id'].to(device)
                    
                    outputs = model(images, targets)
                    
                    if isinstance(outputs, dict) and 'loss' in outputs:
                        loss = outputs['loss']
                        
                        if not (torch.isnan(loss) or torch.isinf(loss)):
                            loss_value = loss.item()
                            val_loss += loss_value
                            val_batch_losses.append(loss_value)
                            num_val_batches += 1
                            val_successful_batches += 1
                        else:
                            logger.warning(f"Invalid validation loss at batch {i}: {loss.item()}")
                            val_failed_batches += 1
                    else:
                        val_failed_batches += 1
                        
                except Exception as e:
                    logger.error(f"Error in validation batch {i}: {e}")
                    val_failed_batches += 1
                    continue
                
                progress.update(val_task, advance=1)
    
    avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0
    
    return {
        'avg_loss': avg_val_loss,
        'batch_losses': val_batch_losses,
        'successful_batches': val_successful_batches,
        'failed_batches': val_failed_batches
    }


def compute_and_log_metrics(model, train_loader, val_loader, device, epoch, train_stats, val_stats, 
                           best_val_iou, best_epoch, best_model_state, wandb_api_key, epoch_start_time,
                           compute_train_metrics=True):
    """Compute metrics and log to console and W&B"""
    # Compute training metrics (only when requested, and limited to 100 samples)
    train_metrics = None
    if compute_train_metrics:
        console.print("[yellow]🔍 Computing training metrics...[/yellow]")
        logger.info("Computing training detection metrics (limited to 100 samples)")
        train_metrics = compute_detection_metrics(model, train_loader, device, num_samples=100)
        
        console.print(f"[green]📊 Train Metrics - IoU:[/green] {train_metrics['mean_iou']:.3f} | "
                    f"[green]Precision:[/green] {train_metrics['precision']:.3f} | "
                    f"[green]Recall:[/green] {train_metrics['recall']:.3f} | "
                    f"[green]F1 Score:[/green] {train_metrics['f1_score']:.3f}")
        
        logger.info(f"Train Metrics - IoU: {train_metrics['mean_iou']:.3f}, "
                   f"Precision: {train_metrics['precision']:.3f}, "
                   f"Recall: {train_metrics['recall']:.3f}, "
                   f"F1: {train_metrics['f1_score']:.3f}")
    
    # Run comprehensive evaluation on validation set
    console.print("[yellow]🔍 Running validation evaluation...[/yellow]")
    logger.info("Computing comprehensive evaluation metrics on validation set")
    
    val_metrics = compute_detection_metrics(model, val_loader, device, num_samples=None)
    
    # Display both training and validation metrics together
    console.print(f"\n[bold cyan]📊 Epoch {epoch+1} Metrics Summary[/bold cyan]")
    if train_metrics:
        console.print(f"[green]Training   - IoU:[/green] {train_metrics['mean_iou']:.3f} | "
                    f"[green]Precision:[/green] {train_metrics['precision']:.3f} | "
                    f"[green]Recall:[/green] {train_metrics['recall']:.3f} | "
                    f"[green]F1:[/green] {train_metrics['f1_score']:.3f}")
    else:
        console.print(f"[yellow]Training   - Metrics skipped this epoch (computed every 2 epochs)[/yellow]")
    console.print(f"[blue]Validation - IoU:[/blue] {val_metrics['mean_iou']:.3f} | "
                f"[blue]Precision:[/blue] {val_metrics['precision']:.3f} | "
                f"[blue]Recall:[/blue] {val_metrics['recall']:.3f} | "
                f"[blue]F1:[/blue] {val_metrics['f1_score']:.3f}")
    
    console.print(f"[cyan]Val Detection Stats - TP:[/cyan] {val_metrics['true_positives']} | "
                f"[red]FP:[/red] {val_metrics['false_positives']} | "
                f"[yellow]FN:[/yellow] {val_metrics['false_negatives']}")
    
    logger.info(f"Val Metrics - IoU: {val_metrics['mean_iou']:.3f}, "
               f"Precision: {val_metrics['precision']:.3f}, "
               f"Recall: {val_metrics['recall']:.3f}, "
               f"F1: {val_metrics['f1_score']:.3f}")
    
    # Log epoch summary to W&B
    if wandb_api_key:
        epoch_time = time.time() - epoch_start_time
        min_batch_loss = np.min(train_stats['batch_losses']) if train_stats['batch_losses'] else 0
        max_batch_loss = np.max(train_stats['batch_losses']) if train_stats['batch_losses'] else 0
        std_batch_loss = np.std(train_stats['batch_losses']) if train_stats['batch_losses'] else 0
        min_val_batch_loss = np.min(val_stats['batch_losses']) if val_stats['batch_losses'] else 0
        max_val_batch_loss = np.max(val_stats['batch_losses']) if val_stats['batch_losses'] else 0
        std_val_batch_loss = np.std(val_stats['batch_losses']) if val_stats['batch_losses'] else 0
        
        log_dict = {
            "epoch": epoch,
            "train_loss": train_stats['avg_loss'],
            "val_loss": val_stats['avg_loss'],
            "train_batch_loss_min": min_batch_loss,
            "train_batch_loss_max": max_batch_loss,
            "train_batch_loss_std": std_batch_loss,
            "train_batch_loss_avg": train_stats['avg_loss'],
            "val_batch_loss_min": min_val_batch_loss,
            "val_batch_loss_max": max_val_batch_loss,
            "val_batch_loss_std": std_val_batch_loss,
            "val_batch_loss_avg": val_stats['avg_loss'],
            "epoch_time": epoch_time,
            "avg_batch_time": train_stats['avg_batch_time'],
            "samples_per_sec": train_stats['avg_samples_per_sec'],
            "train_success_rate": train_stats['successful_batches'] / (train_stats['successful_batches'] + train_stats['failed_batches']) if (train_stats['successful_batches'] + train_stats['failed_batches']) > 0 else 0,
            "val_success_rate": val_stats['successful_batches'] / (val_stats['successful_batches'] + val_stats['failed_batches']) if (val_stats['successful_batches'] + val_stats['failed_batches']) > 0 else 0,
            "total_train_batches": train_stats['successful_batches'] + train_stats['failed_batches'],
            "total_val_batches": val_stats['successful_batches'] + val_stats['failed_batches'],
            "successful_train_batches": train_stats['successful_batches'],
            "successful_val_batches": val_stats['successful_batches'],
            "val_mean_iou": val_metrics['mean_iou'],
            "val_precision": val_metrics['precision'],
            "val_recall": val_metrics['recall'],
            "val_f1_score": val_metrics['f1_score'],
            "val_true_positives": val_metrics['true_positives'],
            "val_false_positives": val_metrics['false_positives'],
            "val_false_negatives": val_metrics['false_negatives']
        }
        
        # Only log training metrics if they were computed
        if train_metrics:
            log_dict.update({
                "train_mean_iou": train_metrics['mean_iou'],
                "train_precision": train_metrics['precision'],
                "train_recall": train_metrics['recall'],
                "train_f1_score": train_metrics['f1_score'],
                "train_true_positives": train_metrics['true_positives'],
                "train_false_positives": train_metrics['false_positives'],
                "train_false_negatives": train_metrics['false_negatives']
            })
        
        wandb.log(log_dict)
    
    # Track best model
    if val_metrics['mean_iou'] > best_val_iou:
        best_val_iou = val_metrics['mean_iou']
        best_epoch = epoch
        best_model_state = model.state_dict()
        console.print(f"[green]✨ New best model! IoU:[/green] {best_val_iou:.3f}")
        logger.info(f"New best model at epoch {epoch+1} with IoU: {best_val_iou:.3f}")
    
    if wandb_api_key:
        wandb.log({"best_val_iou": best_val_iou, "best_epoch_so_far": best_epoch})
    
    return best_val_iou, best_epoch, best_model_state


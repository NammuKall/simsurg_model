#main.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for SimSurgSkill dataset processing and model training
with COCO format data and comprehensive evaluation
Enhanced with Weights & Biases logging and improved monitoring
"""
import os
import sys
import time
import logging
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import wandb
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
import colorlog

# Load environment variables
load_dotenv()

# Import local modules
from src.models import EfficientDetModel
from src.coco_data_loader import get_coco_data_loaders

# Initialize rich console for beautiful output
console = Console()

# Configure enhanced logging
def setup_logging(log_level=logging.INFO):
    """Setup enhanced logging with colors and rich formatting"""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger, log_file

# Initialize logging
logger, log_file = setup_logging()


def main():
    """Main training function with enhanced logging and W&B integration"""
    
    # Start timing
    start_time = time.time()
    
    # Define paths - update these with your actual paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, os.getenv("DATA_DIR", "data/simsurgskill_2021_dataset"))
    coco_dir = os.path.join(base_dir, os.getenv("COCO_DIR", "data/coco_format"))
    results_dir = os.path.join(base_dir, os.getenv("RESULTS_DIR", "results"))
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Display startup banner
    console.print(Panel.fit(
        "[bold blue]SIMSURGSKILL MODEL TRAINING[/bold blue]\n"
        f"[green]Started at:[/green] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"[green]Log file:[/green] {log_file}",
        title="🚀 Training Session",
        border_style="blue"
    ))
    
    logger.info("Starting SimSurgSkill model training")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"COCO directory: {coco_dir}")
    logger.info(f"Results directory: {results_dir}")
    
    # Initialize Weights & Biases
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        try:
            wandb.login(key=wandb_api_key)
            logger.info("Successfully logged into Weights & Biases")
        except Exception as e:
            logger.warning(f"Failed to login to W&B: {e}")
            wandb_api_key = None
    else:
        logger.warning("WANDB_API_KEY not found in environment variables")
    
    # Initialize W&B run
    if wandb_api_key:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "simsurg-model"),
            entity=os.getenv("WANDB_ENTITY"),
            name=f"simsurg_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "learning_rate": float(os.getenv("LEARNING_RATE", "0.001")),
                "batch_size": int(os.getenv("BATCH_SIZE", "4")),
                "num_epochs": int(os.getenv("NUM_EPOCHS", "10")),
                "model": "EfficientDetModel",
                "dataset": "SimSurgSkill",
                "num_classes": 2
            }
        )
        logger.info("W&B run initialized")
    
    # Use already converted COCO format
    coco_paths = {
        'train_dir': os.path.join(coco_dir, 'train', 'images'),
        'val_dir': os.path.join(coco_dir, 'val', 'images'),
        'test_dir': os.path.join(coco_dir, 'test', 'images'),
        'train_ann': os.path.join(coco_dir, 'annotations', 'instances_train.json'),
        'val_ann': os.path.join(coco_dir, 'annotations', 'instances_val.json'),
        'test_ann': os.path.join(coco_dir, 'annotations', 'instances_test.json')
    }
    
    # Verify COCO files exist with enhanced logging
    console.print("\n[bold blue]Verifying COCO format files...[/bold blue]")
    logger.info("Verifying COCO format files")
    
    file_status_table = Table(title="File Verification Status")
    file_status_table.add_column("File Type", style="cyan")
    file_status_table.add_column("Path", style="magenta")
    file_status_table.add_column("Status", style="green")
    
    all_files_exist = True
    for key, path in coco_paths.items():
        exists = os.path.exists(path)
        status = "✅ Found" if exists else "❌ Missing"
        file_status_table.add_row(key, path, status)
        
        if exists:
            logger.info(f"Found {key}: {path}")
        else:
            logger.error(f"Missing {key}: {path}")
            all_files_exist = False
    
    console.print(file_status_table)
    
    if not all_files_exist:
        console.print("[bold red]❌ Some required files are missing! Please check the paths.[/bold red]")
        logger.error("Training aborted due to missing files")
        return None
    
    # Get data loaders for COCO format data
    console.print("\n[bold blue]Creating data loaders...[/bold blue]")
    logger.info("Creating data loaders")
    
    batch_size = int(os.getenv("BATCH_SIZE", "4"))
    train_loader, val_loader, test_loader = get_coco_data_loaders(coco_paths, batch_size=batch_size)
    
    # Display data loader information
    data_info_table = Table(title="Data Loader Information")
    data_info_table.add_column("Split", style="cyan")
    data_info_table.add_column("Batches", style="green")
    data_info_table.add_column("Total Samples", style="magenta")
    
    data_info_table.add_row("Training", str(len(train_loader)), str(len(train_loader) * batch_size))
    data_info_table.add_row("Validation", str(len(val_loader)), str(len(val_loader) * batch_size))
    data_info_table.add_row("Test", str(len(test_loader)), str(len(test_loader) * batch_size))
    
    console.print(data_info_table)
    
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Initialize model and optimizer
    device_str = os.getenv("DEVICE", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    console.print(f"\n[bold green]🔧 Using device:[/bold green] {device}")
    logger.info(f"Using device: {device}")
    
    # Log device information
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        console.print(f"[green]GPU:[/green] {gpu_name} ({gpu_memory:.1f} GB)")
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        if wandb_api_key:
            wandb.config.update({
                "device": str(device),
                "gpu_name": gpu_name,
                "gpu_memory_gb": gpu_memory
            })
    else:
        console.print(f"[yellow]CPU Training Mode[/yellow] - Consider reducing batch size for better performance")
        logger.info("CPU training mode - batch size may need adjustment for optimal performance")
        
        if wandb_api_key:
            wandb.config.update({
                "device": str(device),
                "training_mode": "cpu"
            })
    
    model = EfficientDetModel(num_classes=2).to(device)
    learning_rate = float(os.getenv("LEARNING_RATE", "0.001"))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    console.print(f"[bold green]✅ Model initialized:[/bold green] {total_params:,} total parameters, {trainable_params:,} trainable")
    logger.info(f"Model initialized with {total_params:,} total parameters, {trainable_params:,} trainable")
    
    if wandb_api_key:
        wandb.config.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_architecture": "EfficientDetModel"
        })
    
    # Performance tracking
    performance_history = {
        'train_loss': [],
        'val_loss': [],
        'avg_batch_time': [],
        'samples_per_sec': []
    }
    
    # Training loop
    num_epochs = int(os.getenv("NUM_EPOCHS", "10"))
    train_losses = []
    val_losses = []
    
    console.print(Panel.fit(
        f"[bold green]STARTING TRAINING[/bold green]\n"
        f"Epochs: {num_epochs} | Batch Size: {batch_size} | Learning Rate: {learning_rate}",
        title="🚀 Training Configuration",
        border_style="green"
    ))
    
    logger.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        batch_times = []
        
        console.print(f"\n[bold blue]📊 Epoch [{epoch+1}/{num_epochs}][/bold blue]")
        logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        successful_batches = 0
        failed_batches = 0
        
        # Create progress bar for training
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
                    # Move data to device
                    images = images.to(device)
                    
                    # Move targets to device
                    for j in range(len(targets)):
                        targets[j]['boxes'] = targets[j]['boxes'].to(device)
                        targets[j]['labels'] = targets[j]['labels'].to(device)
                        targets[j]['image_id'] = targets[j]['image_id'].to(device)
                    
                    # Forward pass
                    outputs = model(images, targets)
                    
                    # Extract loss from output dictionary
                    if isinstance(outputs, dict) and 'loss' in outputs:
                        loss = outputs['loss']
                        
                        # Check for invalid loss values
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning(f"Invalid loss value at batch {i}: {loss.item()}")
                            failed_batches += 1
                            continue
                        
                        # Backward and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        
                        # Check for gradient issues
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
                        
                        train_loss += loss.item()
                        num_batches += 1
                        successful_batches += 1
                        
                        # Track batch performance
                        batch_time = time.time() - batch_start_time
                        batch_times.append(batch_time)
                        samples_per_sec = batch_size / batch_time if batch_time > 0 else 0
                        
                        # Log to W&B
                        if wandb_api_key and i % 10 == 0:
                            wandb.log({
                                "epoch": epoch,
                                "batch": i,
                                "train_loss": loss.item(),
                                "grad_norm": grad_norm,
                                "learning_rate": optimizer.param_groups[0]['lr'],
                                "batch_time": batch_time,
                                "samples_per_sec": samples_per_sec
                            })
                        
                        if (i + 1) % 10 == 0:
                            logger.info(f'Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, '
                                       f'Grad Norm: {grad_norm:.4f}, Batch Time: {batch_time:.3f}s, '
                                       f'Speed: {samples_per_sec:.1f} samples/sec')
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
        train_losses.append(avg_train_loss)
        
        # Performance metrics
        avg_batch_time = np.mean(batch_times) if batch_times else 0
        avg_samples_per_sec = np.mean([batch_size / t for t in batch_times if t > 0]) if batch_times else 0
        performance_history['avg_batch_time'].append(avg_batch_time)
        performance_history['samples_per_sec'].append(avg_samples_per_sec)
        
        # Log epoch training results with performance
        console.print(f"[green]📈 Training Loss:[/green] {avg_train_loss:.4f} | "
                    f"[green]Successful:[/green] {successful_batches} | "
                    f"[red]Failed:[/red] {failed_batches} | "
                    f"[yellow]Speed:[/yellow] {avg_samples_per_sec:.1f} samples/sec")
        
        logger.info(f"Epoch {epoch+1} training completed - Loss: {avg_train_loss:.4f}, "
                   f"Successful: {successful_batches}, Failed: {failed_batches}, "
                   f"Avg Batch Time: {avg_batch_time:.3f}s, Speed: {avg_samples_per_sec:.1f} samples/sec")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        val_successful_batches = 0
        val_failed_batches = 0
        
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
                        
                        # Move targets to device
                        for j in range(len(targets)):
                            targets[j]['boxes'] = targets[j]['boxes'].to(device)
                            targets[j]['labels'] = targets[j]['labels'].to(device)
                            targets[j]['image_id'] = targets[j]['image_id'].to(device)
                        
                        # Forward pass
                        outputs = model(images, targets)
                        
                        # Extract loss
                        if isinstance(outputs, dict) and 'loss' in outputs:
                            loss = outputs['loss']
                            
                            # Check for invalid loss values
                            if not (torch.isnan(loss) or torch.isinf(loss)):
                                val_loss += loss.item()
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
        val_losses.append(avg_val_loss)
        
        # Log epoch validation results
        console.print(f"[blue]📉 Validation Loss:[/blue] {avg_val_loss:.4f} | "
                    f"[green]Successful Batches:[/green] {val_successful_batches} | "
                    f"[red]Failed Batches:[/red] {val_failed_batches}")
        
        logger.info(f"Epoch {epoch+1} validation completed - Loss: {avg_val_loss:.4f}, "
                   f"Successful: {val_successful_batches}, Failed: {val_failed_batches}")
        
        # Log epoch summary to W&B
        if wandb_api_key:
            epoch_time = time.time() - epoch_start_time
            wandb.log({
                "epoch": epoch,
                "epoch_train_loss": avg_train_loss,
                "epoch_val_loss": avg_val_loss,
                "epoch_time": epoch_time,
                "train_success_rate": successful_batches / (successful_batches + failed_batches) if (successful_batches + failed_batches) > 0 else 0,
                "val_success_rate": val_successful_batches / (val_successful_batches + val_failed_batches) if (val_successful_batches + val_failed_batches) > 0 else 0,
                "avg_batch_time": avg_batch_time,
                "samples_per_sec": avg_samples_per_sec
            })
        
        # Early stopping check (optional)
        if len(val_losses) > 3:
            recent_val_losses = val_losses[-3:]
            if all(recent_val_losses[i] >= recent_val_losses[i+1] for i in range(len(recent_val_losses)-1)):
                logger.info("Validation loss is increasing - consider early stopping")
    
    # Save the model
    console.print(Panel.fit(
        "[bold green]SAVING MODEL[/bold green]",
        title="💾 Model Saving",
        border_style="green"
    ))
    
    model_save_dir = os.path.join(base_dir, os.getenv("MODEL_SAVE_DIR", "models"))
    os.makedirs(model_save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(model_save_dir, f'model_{timestamp}.pth')
    
    # Save model state dict and additional training info
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epoch': num_epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'timestamp': timestamp
    }
    
    torch.save(save_dict, model_save_path)
    console.print(f"[green]✅ Model saved to:[/green] {model_save_path}")
    logger.info(f"Model saved to: {model_save_path}")
    
    # Log model to W&B
    if wandb_api_key:
        wandb.save(model_save_path)
        logger.info("Model uploaded to W&B")
    
    # Generate training plots
    console.print(Panel.fit(
        "[bold blue]GENERATING TRAINING PLOTS[/bold blue]",
        title="📊 Visualization",
        border_style="blue"
    ))
    
    plt.figure(figsize=(15, 6))
    
    # Training and validation loss
    plt.subplot(1, 3, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    plt.plot(range(1, num_epochs + 1), val_losses, 'r-o', label='Validation Loss', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Loss statistics
    plt.subplot(1, 3, 2)
    metrics_names = ['Final Train\nLoss', 'Final Val\nLoss', 'Min Train\nLoss', 'Min Val\nLoss']
    metrics_values = [
        train_losses[-1] if train_losses else 0,
        val_losses[-1] if val_losses else 0,
        min(train_losses) if train_losses else 0,
        min(val_losses) if val_losses else 0
    ]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Training Summary', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=9)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Training progress
    plt.subplot(1, 3, 3)
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
    plt.fill_between(epochs, train_losses, alpha=0.3, color='blue')
    plt.fill_between(epochs, val_losses, alpha=0.3, color='red')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss Progress', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    training_plots_path = os.path.join(results_dir, f'training_summary_{timestamp}.png')
    plt.savefig(training_plots_path, dpi=300, bbox_inches='tight')
    console.print(f"[green]✅ Training plots saved to:[/green] {training_plots_path}")
    logger.info(f"Training plots saved to: {training_plots_path}")
    
    # Log plots to W&B
    if wandb_api_key:
        wandb.log({"training_plots": wandb.Image(training_plots_path)})
    
    plt.show()
    
    # Final summary
    total_time = time.time() - start_time
    
    console.print(Panel.fit(
        f"[bold green]TRAINING COMPLETE![/bold green]\n"
        f"[green]Total Time:[/green] {total_time/60:.1f} minutes\n"
        f"[green]Final Training Loss:[/green] {train_losses[-1]:.4f}\n"
        f"[green]Final Validation Loss:[/green] {val_losses[-1]:.4f}\n"
        f"[green]Best Training Loss:[/green] {min(train_losses):.4f} (Epoch {train_losses.index(min(train_losses))+1})\n"
        f"[green]Best Validation Loss:[/green] {min(val_losses):.4f} (Epoch {val_losses.index(min(val_losses))+1})",
        title="🎉 Training Complete",
        border_style="green"
    ))
    
    # Create final results table
    results_table = Table(title="Final Training Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    
    results_table.add_row("Total Training Time", f"{total_time/60:.1f} minutes")
    results_table.add_row("Final Training Loss", f"{train_losses[-1]:.4f}")
    results_table.add_row("Final Validation Loss", f"{val_losses[-1]:.4f}")
    results_table.add_row("Best Training Loss", f"{min(train_losses):.4f}")
    results_table.add_row("Best Validation Loss", f"{min(val_losses):.4f}")
    results_table.add_row("Model Parameters", f"{total_params:,}")
    results_table.add_row("Epochs Completed", str(num_epochs))
    
    console.print(results_table)
    
    logger.info("Training completed successfully")
    logger.info(f"Total training time: {total_time/60:.1f} minutes")
    logger.info(f"Final training loss: {train_losses[-1]:.4f}")
    logger.info(f"Final validation loss: {val_losses[-1]:.4f}")
    
    # Log final metrics to W&B
    if wandb_api_key:
        wandb.log({
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "best_train_loss": min(train_losses),
            "best_val_loss": min(val_losses),
            "total_training_time": total_time,
            "total_epochs": num_epochs
        })
        wandb.finish()
        logger.info("W&B run finished")
    
    console.print(f"\n[bold blue]📁 Outputs saved to:[/bold blue]")
    console.print(f"  • Model: {model_save_path}")
    console.print(f"  • Training plots: {training_plots_path}")
    console.print(f"  • Log file: {log_file}")
    
    console.print(f"\n[bold blue]🎯 Next steps:[/bold blue]")
    console.print(f"  1. Review training plots in {results_dir}")
    console.print(f"  2. Run evaluation on test set")
    console.print(f"  3. Fine-tune hyperparameters if needed")
    console.print(f"  4. Check W&B dashboard for detailed metrics")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_path': model_save_path,
        'plots_path': training_plots_path,
        'log_file': str(log_file),
        'total_time': total_time,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    }


if __name__ == "__main__":
    results = main()

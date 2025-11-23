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
from src.training import train_epoch, validate_epoch, compute_and_log_metrics
from src.testing import compute_detection_metrics

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


def setup_paths_and_directories():
    """Setup and create necessary directories"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, os.getenv("DATA_DIR", "data/simsurgskill_2021_dataset"))
    coco_dir = os.path.join(base_dir, os.getenv("COCO_DIR", "data/coco_format"))
    results_dir = os.path.join(base_dir, os.getenv("RESULTS_DIR", "results"))
    
    os.makedirs(results_dir, exist_ok=True)
    
    coco_paths = {
        'train_dir': os.path.join(coco_dir, 'train', 'images'),
        'val_dir': os.path.join(coco_dir, 'val', 'images'),
        'test_dir': os.path.join(coco_dir, 'test', 'images'),
        'train_ann': os.path.join(coco_dir, 'annotations', 'instances_train.json'),
        'val_ann': os.path.join(coco_dir, 'annotations', 'instances_val.json'),
        'test_ann': os.path.join(coco_dir, 'annotations', 'instances_test.json')
    }
    
    return {
        'base_dir': base_dir,
        'data_dir': data_dir,
        'coco_dir': coco_dir,
        'results_dir': results_dir,
        'coco_paths': coco_paths
    }


def initialize_wandb():
    """Initialize Weights & Biases"""
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
                "num_classes": 2,
                "optimizer": "Adam",
                "loss_function": "CrossEntropy + SmoothL1Loss",
                "evaluation_frequency": "Every 100 batches + end of epoch"
            }
        )
        logger.info("W&B run initialized")
    
    return wandb_api_key


def verify_coco_files(coco_paths):
    """Verify COCO format files exist"""
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
        return False
    
    return True


def setup_data_loaders(coco_paths, batch_size):
    """Setup data loaders and display information"""
    console.print("\n[bold blue]Creating data loaders...[/bold blue]")
    logger.info("Creating data loaders")
    
    train_loader, val_loader, test_loader = get_coco_data_loaders(coco_paths, batch_size=batch_size)
    
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
    
    return train_loader, val_loader, test_loader


def setup_device_and_model(wandb_api_key):
    """Setup device, model, and optimizer"""
    device_str = os.getenv("DEVICE", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    console.print(f"\n[bold green]🔧 Using device:[/bold green] {device}")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        console.print(f"[green]GPU:[/green] {gpu_name} ({gpu_memory:.1f} GB)")
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        if wandb_api_key:
            wandb.config.update({
                "device": str(device),
                "gpu_name": gpu_name,
                "gpu_memory_gb": gpu_memory,
                "cuda_version": torch.version.cuda,
                "torch_version": torch.__version__
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
    
    return device, model, optimizer, learning_rate, total_params


def save_model(model, best_model_state, optimizer, train_losses, val_losses, best_val_iou, 
              best_epoch, num_epochs, learning_rate, batch_size, base_dir, wandb_api_key):
    """Save model and upload to W&B if enabled"""
    console.print(Panel.fit(
        "[bold green]SAVING MODEL[/bold green]",
        title="💾 Model Saving",
        border_style="green"
    ))
    
    model_save_dir = os.path.join(base_dir, os.getenv("MODEL_SAVE_DIR", "models"))
    os.makedirs(model_save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(model_save_dir, f'model_{timestamp}.pth')
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'best_model_state': best_model_state if best_model_state else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_iou': best_val_iou,
        'best_epoch': best_epoch,
        'epoch': num_epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'timestamp': timestamp
    }
    
    torch.save(save_dict, model_save_path)
    
    best_model_path = None
    if best_model_state and any(not torch.equal(best_model_state[k], model.state_dict()[k]) for k in best_model_state.keys()):
        best_model_path = os.path.join(model_save_dir, f'best_model_iou{best_val_iou:.3f}_epoch{best_epoch+1}_{timestamp}.pth')
        torch.save({
            'model_state_dict': best_model_state,
            'best_val_iou': best_val_iou,
            'best_epoch': best_epoch
        }, best_model_path)
        console.print(f"[green]✅ Best model saved to:[/green] {best_model_path}")
        logger.info(f"Best model saved to: {best_model_path}")
    
    console.print(f"[green]✅ Model saved to:[/green] {model_save_path}")
    logger.info(f"Model saved to: {model_save_path}")
    
    if wandb_api_key:
        wandb.save(model_save_path)
        logger.info("Model uploaded to W&B")
    
    return model_save_path, best_model_path


def generate_training_plots(train_losses, val_losses, num_epochs, results_dir, timestamp, wandb_api_key):
    """Generate and save training plots"""
    console.print(Panel.fit(
        "[bold blue]GENERATING TRAINING PLOTS[/bold blue]",
        title="📊 Visualization",
        border_style="blue"
    ))
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    plt.plot(range(1, num_epochs + 1), val_losses, 'r-o', label='Validation Loss', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
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
    
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
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
    
    if wandb_api_key:
        wandb.log({"training_plots": wandb.Image(training_plots_path)})
    
    plt.show()
    
    return training_plots_path


def display_final_summary(train_losses, val_losses, total_time, total_params, num_epochs, 
                        best_val_iou, best_epoch, model_save_path, training_plots_path, 
                        log_file, results_dir, wandb_api_key):
    """Display final summary and log to W&B"""
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
    
    if wandb_api_key:
        wandb.log({
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "best_train_loss": min(train_losses),
            "best_val_loss": min(val_losses),
            "best_val_iou_final": best_val_iou,
            "best_epoch_final": best_epoch,
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


def main():
    """Main training function - orchestrates the training pipeline"""
    
    start_time = time.time()
    
    # Display startup banner
    console.print(Panel.fit(
        "[bold blue]SIMSURGSKILL MODEL TRAINING[/bold blue]\n"
        f"[green]Started at:[/green] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"[green]Log file:[/green] {log_file}",
        title="🚀 Training Session",
        border_style="blue"
    ))
    
    # Setup phase
    paths = setup_paths_and_directories()
    logger.info("Starting SimSurgSkill model training")
    logger.info(f"Base directory: {paths['base_dir']}")
    logger.info(f"Data directory: {paths['data_dir']}")
    logger.info(f"COCO directory: {paths['coco_dir']}")
    logger.info(f"Results directory: {paths['results_dir']}")
    
    wandb_api_key = initialize_wandb()
    
    if not verify_coco_files(paths['coco_paths']):
        return None
    
    batch_size = int(os.getenv("BATCH_SIZE", "4"))
    train_loader, val_loader, test_loader = setup_data_loaders(paths['coco_paths'], batch_size)
    
    device, model, optimizer, learning_rate, total_params = setup_device_and_model(wandb_api_key)
    
    # Training loop
    num_epochs = int(os.getenv("NUM_EPOCHS", "10"))
    train_losses = []
    val_losses = []
    best_val_iou = 0.0
    best_epoch = 0
    best_model_state = None
    
    console.print(Panel.fit(
        f"[bold green]STARTING TRAINING[/bold green]\n"
        f"Epochs: {num_epochs} | Batch Size: {batch_size} | Learning Rate: {learning_rate}",
        title="🚀 Training Configuration",
        border_style="green"
    ))
    
    logger.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        console.print(f"\n[bold blue]📊 Epoch [{epoch+1}/{num_epochs}][/bold blue]")
        logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
        
        # Train epoch
        train_stats = train_epoch(model, train_loader, optimizer, device, epoch, num_epochs, 
                                 wandb_api_key, batch_size, val_loader)
        train_losses.append(train_stats['avg_loss'])
        
        # Log training results
        console.print(f"[green]📈 Training Loss:[/green] {train_stats['avg_loss']:.4f} | "
                    f"[green]Successful:[/green] {train_stats['successful_batches']} | "
                    f"[red]Failed:[/red] {train_stats['failed_batches']} | "
                    f"[yellow]Speed:[/yellow] {train_stats['avg_samples_per_sec']:.1f} samples/sec")
        
        logger.info(f"Epoch {epoch+1} training completed - Loss: {train_stats['avg_loss']:.4f}, "
                   f"Successful: {train_stats['successful_batches']}, Failed: {train_stats['failed_batches']}, "
                   f"Avg Batch Time: {train_stats['avg_batch_time']:.3f}s, "
                   f"Speed: {train_stats['avg_samples_per_sec']:.1f} samples/sec")
        
        # Validate epoch
        val_stats = validate_epoch(model, val_loader, device, epoch, num_epochs)
        val_losses.append(val_stats['avg_loss'])
        
        # Log validation results
        console.print(f"[blue]📉 Validation Loss:[/blue] {val_stats['avg_loss']:.4f} | "
                    f"[green]Successful Batches:[/green] {val_stats['successful_batches']} | "
                    f"[red]Failed Batches:[/red] {val_stats['failed_batches']}")
        
        logger.info(f"Epoch {epoch+1} validation completed - Loss: {val_stats['avg_loss']:.4f}, "
                   f"Successful: {val_stats['successful_batches']}, Failed: {val_stats['failed_batches']}")
        
        # Compute metrics and update best model
        # Compute training metrics every 2 epochs to reduce overhead
        compute_train_metrics = (epoch % 2 == 0)
        best_val_iou, best_epoch, best_model_state = compute_and_log_metrics(
            model, train_loader, val_loader, device, epoch, train_stats, val_stats,
            best_val_iou, best_epoch, best_model_state, wandb_api_key, epoch_start_time,
            compute_train_metrics=compute_train_metrics
        )
        
        # Early stopping check (optional)
        if len(val_losses) > 3:
            recent_val_losses = val_losses[-3:]
            if all(recent_val_losses[i] >= recent_val_losses[i+1] for i in range(len(recent_val_losses)-1)):
                logger.info("Validation loss is increasing - consider early stopping")
    
    # Post-training phase
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path, best_model_path = save_model(
        model, best_model_state, optimizer, train_losses, val_losses, best_val_iou,
        best_epoch, num_epochs, learning_rate, batch_size, paths['base_dir'], wandb_api_key
    )
    
    training_plots_path = generate_training_plots(
        train_losses, val_losses, num_epochs, paths['results_dir'], timestamp, wandb_api_key
    )
    
    total_time = time.time() - start_time
    results = display_final_summary(
        train_losses, val_losses, total_time, total_params, num_epochs, best_val_iou, best_epoch,
        model_save_path, training_plots_path, log_file, paths['results_dir'], wandb_api_key
    )
    
    return results


if __name__ == "__main__":
    results = main()

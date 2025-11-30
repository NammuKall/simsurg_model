# SimSurg Model - Multi-Model Object Detection Training Pipeline

This project provides a comprehensive object detection model training pipeline for the SimSurgSkill dataset with support for multiple model architectures, comprehensive logging, monitoring, and Weights & Biases integration.

## 🚀 Features

- **Multiple Model Architectures**: Support for EfficientDet, FasterRCNN, YOLOv5, and ResNet models
- **Model Factory Pattern**: Easy switching between models via configuration
- **Model Variants**: Pre-configured variants for each model (e.g., YOLOv5-Small, FasterRCNN-Frozen)
- **Enhanced Logging**: Beautiful console output with Rich library and comprehensive file logging
- **Weights & Biases Integration**: Real-time experiment tracking and visualization
- **Robust Error Handling**: Detailed error tracking and recovery mechanisms
- **Progress Monitoring**: Real-time progress bars and training metrics
- **Model Checkpointing**: Automatic model saving with metadata
- **Visualization**: Comprehensive training plots and metrics
- **Inference Script**: Standalone inference script for model evaluation
- **COCO Format Support**: Full support for COCO-format datasets

## 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Weights & Biases account (optional but recommended)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd simsurg_model
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   # Copy the example environment file
   cp env.example .env
   
   # Edit .env with your actual values
   nano .env
   ```

## 🔧 Configuration

### Environment Variables (.env file)

```bash
# Weights & Biases Configuration
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=simsurg-model
WANDB_ENTITY=your_wandb_username

# Training Configuration
BATCH_SIZE=4
LEARNING_RATE=0.001
NUM_EPOCHS=10
DEVICE=auto
NUM_WORKERS=2

# Model Configuration
MODEL_NAME=EfficientDet
# Options: EfficientDet, ResNet, FasterRCNN, YOLOv5
MODEL_VARIANT=EfficientDet-Small  # Optional: specify model variant

# Data Paths
DATA_DIR=data/simsurgskill_2021_dataset
COCO_DIR=data/coco_format
RESULTS_DIR=results
MODEL_SAVE_DIR=models
```

### Available Models

#### EfficientDet
- **Description**: EfficientDet-style model with ResNet50 backbone and feature fusion
- **Variants**: `EfficientDet-Small`, `EfficientDet-Large`
- **Training Metrics**: Computed every epoch
- **Default Parameters**: `num_classes=2`, `num_anchors=9`

#### FasterRCNN
- **Description**: Faster R-CNN with ResNet50-FPN backbone (torchvision implementation)
- **Variants**: `FasterRCNN-Frozen`, `FasterRCNN-Full`
- **Training Metrics**: Computed every 2 epochs
- **Default Parameters**: `num_classes=2`, `pretrained=True`, `trainable_backbone_layers=3`

#### YOLOv5
- **Description**: YOLOv5 model with CSPDarkNet backbone and PANet
- **Variants**: `YOLOv5-Nano`, `YOLOv5-Small`, `YOLOv5-Medium`, `YOLOv5-Large`, `YOLOv5-XLarge`, `YOLOv5-Focal`, `YOLOv5-Smooth`
- **Training Metrics**: Computed every 2 epochs
- **Special Features**: 
  - Differential learning rates (lower for backbone, higher for detection heads)
  - Cosine annealing scheduler with warmup
  - Optimized loss weights
- **Default Parameters**: `num_classes=2`, `pretrained=True`, `model_size="s"`, `input_size=640`

#### ResNet
- **Description**: ResNet model for classification
- **Variants**: `ResNet-18`, `ResNet-34`
- **Training Metrics**: Computed every 2 epochs
- **Default Parameters**: `num_classes=2`, `layers=[2, 2, 2, 2]`

### Getting Your W&B API Key

1. Go to [wandb.ai](https://wandb.ai)
2. Sign up or log in
3. Go to your profile settings
4. Copy your API key
5. Add it to your `.env` file

## 📁 Project Structure

```
simsurg_model/
├── main.py                 # Main training script
├── inference.py            # Inference script for model evaluation
├── data.py                 # Data processing pipeline
├── requirements.txt        # Dependencies
├── env.example            # Environment variables template
├── README.md              # This file
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_factory.py    # Model factory for creating models
│   │   ├── model_config.py     # Model configuration definitions
│   │   ├── efficientdet.py     # EfficientDet model
│   │   ├── faster_rcnn.py      # Faster R-CNN model
│   │   ├── yolov5.py           # YOLOv5 model
│   │   ├── resnet.py           # ResNet model
│   │   └── README.md           # Model documentation
│   ├── coco_data_loader.py     # COCO format data loading
│   ├── coco_converter.py        # COCO format conversion utilities
│   ├── training.py              # Training functions
│   ├── testing.py               # Testing and evaluation functions
│   ├── evaluation_metrics.py   # Evaluation metrics computation
│   ├── visualization.py        # Visualization utilities
│   └── utils.py                 # Utility functions
├── logs/                  # Training logs (auto-created)
├── models/                # Saved models (auto-created)
└── results/               # Training plots and results (auto-created)
```

## 🏃‍♂️ Running Training

### Basic Usage

```bash
python main.py
```

This will use the default model (EfficientDet) and settings from your `.env` file.

### Training Different Models

You can switch between models using the `MODEL_NAME` environment variable:

```bash
# Train EfficientDet
MODEL_NAME=EfficientDet python main.py

# Train FasterRCNN
MODEL_NAME=FasterRCNN python main.py

# Train YOLOv5
MODEL_NAME=YOLOv5 python main.py

# Train ResNet
MODEL_NAME=ResNet python main.py
```

### Using Model Variants

You can specify model variants for additional configuration:

```bash
# EfficientDet with small variant
MODEL_NAME=EfficientDet MODEL_VARIANT=EfficientDet-Small python main.py

# FasterRCNN with frozen backbone
MODEL_NAME=FasterRCNN MODEL_VARIANT=FasterRCNN-Frozen python main.py

# YOLOv5 with focal loss
MODEL_NAME=YOLOv5 MODEL_VARIANT=YOLOv5-Focal python main.py
```

### Custom Configuration

You can override any setting by modifying your `.env` file or setting environment variables:

```bash
# Example: Run with different batch size
BATCH_SIZE=8 python main.py

# Example: Run for more epochs
NUM_EPOCHS=20 python main.py

# Example: Combine multiple settings
MODEL_NAME=YOLOv5 MODEL_VARIANT=YOLOv5-Small BATCH_SIZE=4 NUM_EPOCHS=50 python main.py
```

## 🔍 Running Inference

After training, you can run inference on test images:

```bash
python inference.py --model models/model_YYYYMMDD_HHMMSS.pth
```

The inference script will:
- Load the trained model checkpoint
- Run inference on test/validation images
- Visualize predictions with bounding boxes
- Compute detection metrics (IoU, precision, recall, mAP)
- Generate detailed reports and statistics
- Save visualized results

For more details, see the documentation at the top of `inference.py`.

## 📊 Monitoring Training

### Console Output

The enhanced logging provides:
- **Rich Console Output**: Beautiful, colored progress bars and status updates
- **Real-time Metrics**: Live loss tracking and batch progress
- **Error Handling**: Detailed error messages with stack traces
- **Summary Tables**: Comprehensive training statistics
- **Model-Specific Information**: Displays selected model and variant

### Training Metrics Behavior

- **EfficientDet**: Training metrics are computed every epoch
- **Other Models**: Training metrics are computed every 2 epochs to reduce overhead
- **Validation Metrics**: Always computed every epoch for all models
- **Periodic Evaluation**: Every 100 batches during training (validation set, 50 samples)

### Weights & Biases Dashboard

If W&B is configured, you'll get:
- **Real-time Metrics**: Loss curves, learning rate, gradient norms
- **System Monitoring**: GPU usage, memory consumption
- **Model Artifacts**: Automatic model checkpointing
- **Experiment Comparison**: Compare different runs
- **Model-Specific Metrics**: Training and validation IoU, precision, recall, F1 scores

### Log Files

Detailed logs are saved to `logs/training_YYYYMMDD_HHMMSS.log` with:
- Timestamped entries
- Debug information
- Error details
- Training progress
- Model configuration

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `BATCH_SIZE` in `.env`
   - Use `DEVICE=cpu` for CPU-only training
   - Try a smaller model variant (e.g., `YOLOv5-Nano`)

2. **W&B Login Issues**:
   - Verify your API key is correct
   - Check internet connection
   - Ensure W&B project exists

3. **Data Loading Errors**:
   - Verify COCO format files exist
   - Check file paths in `.env`
   - Ensure proper permissions
   - Run `python data.py` to generate COCO format if needed

4. **Model Loading Errors**:
   - Ensure `MODEL_NAME` matches the checkpoint's model architecture
   - Check that checkpoint file exists and is valid
   - Verify `MODEL_VARIANT` matches the training configuration

### Error Analysis

The enhanced logging will help identify issues:

- **Invalid Loss Values**: NaN or infinite losses are detected and logged
- **Gradient Issues**: Gradient norm monitoring helps identify training instability
- **Batch Failures**: Detailed tracking of successful vs failed batches
- **Memory Issues**: GPU memory monitoring and warnings
- **Model-Specific Issues**: YOLOv5 includes gradient clipping and differential learning rates

## 📈 Model-Specific Features

### YOLOv5 Optimizations

YOLOv5 includes several optimizations:
- **Differential Learning Rates**: Lower LR for pretrained backbone (10x reduction), full LR for detection heads
- **Learning Rate Scheduler**: Cosine annealing with warmup (10% of total epochs)
- **Gradient Clipping**: Prevents exploding gradients (max norm: 10.0)
- **Optimized Loss Weights**: Tuned for better convergence
- **Label Smoothing**: Available via variants

### EfficientDet Features

- **Training Metrics**: Computed every epoch (more frequent than other models)
- **Feature Fusion**: Multi-scale feature fusion architecture
- **Anchor-based Detection**: Configurable number of anchors

### FasterRCNN Features

- **Backbone Control**: Configurable trainable backbone layers
- **Pretrained Weights**: Uses torchvision pretrained ResNet50-FPN
- **Frozen Variant**: Option to freeze backbone completely

## 🎯 Next Steps

1. **Review Training Plots**: Check the generated plots in `results/`
2. **Monitor W&B Dashboard**: Analyze training curves and metrics
3. **Evaluate Model**: Run inference on test set using `inference.py`
4. **Hyperparameter Tuning**: Adjust learning rate, batch size, etc.
5. **Model Comparison**: Try different models and variants
6. **Model Optimization**: Consider different architectures or loss functions

## 📝 Logging Levels

You can adjust logging verbosity by modifying the `setup_logging()` call in `main.py`:

```python
# For more detailed logging
logger, log_file = setup_logging(log_level=logging.DEBUG)

# For less verbose logging
logger, log_file = setup_logging(log_level=logging.WARNING)
```

## 📚 Additional Documentation

- **Model Documentation**: See `src/models/README.md` for detailed model information
- **Refactoring Summary**: See `REFACTORING_SUMMARY.md` for architecture details
- **YOLOv5 Issues**: See `YOLOV5_ISSUES.md` for YOLOv5-specific notes
- **Optimization Plans**: See `MODEL_OPTIMIZATION_PLAN.md` and `QUICK_START_OPTIMIZATIONS.md`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- SimSurgSkill dataset creators
- PyTorch team
- Weights & Biases team
- Rich library contributors
- Ultralytics (YOLOv5)

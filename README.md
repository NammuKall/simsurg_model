# SimSurg Model - Enhanced Training with Weights & Biases

This project provides an enhanced object detection model training pipeline for the SimSurgSkill dataset with comprehensive logging, monitoring, and Weights & Biases integration.

## 🚀 Features

- **Enhanced Logging**: Beautiful console output with Rich library and comprehensive file logging
- **Weights & Biases Integration**: Real-time experiment tracking and visualization
- **Robust Error Handling**: Detailed error tracking and recovery mechanisms
- **Progress Monitoring**: Real-time progress bars and training metrics
- **Model Checkpointing**: Automatic model saving with metadata
- **Visualization**: Comprehensive training plots and metrics

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

# Data Paths
DATA_DIR=data/simsurgskill_2021_dataset
COCO_DIR=data/coco_format
RESULTS_DIR=results
MODEL_SAVE_DIR=models
```

### Getting Your W&B API Key

1. Go to [wandb.ai](https://wandb.ai)
2. Sign up or log in
3. Go to your profile settings
4. Copy your API key
5. Add it to your `.env` file

## 📁 Project Structure

```
simsurg_model/
├── main.py                 # Enhanced training script
├── requirements.txt        # Dependencies
├── env.example            # Environment variables template
├── README.md              # This file
├── src/
│   ├── models.py          # Model definitions
│   ├── coco_data_loader.py # Data loading utilities
│   └── ...                # Other source files
├── logs/                  # Training logs (auto-created)
├── models/                # Saved models (auto-created)
└── results/               # Training plots and results (auto-created)
```

## 🏃‍♂️ Running Training

### Basic Usage

```bash
python main.py
```

### With Custom Configuration

You can override any setting by modifying your `.env` file or setting environment variables:

```bash
# Example: Run with different batch size
BATCH_SIZE=8 python main.py

# Example: Run for more epochs
NUM_EPOCHS=20 python main.py
```

## 📊 Monitoring Training

### Console Output

The enhanced logging provides:
- **Rich Console Output**: Beautiful, colored progress bars and status updates
- **Real-time Metrics**: Live loss tracking and batch progress
- **Error Handling**: Detailed error messages with stack traces
- **Summary Tables**: Comprehensive training statistics

### Weights & Biases Dashboard

If W&B is configured, you'll get:
- **Real-time Metrics**: Loss curves, learning rate, gradient norms
- **System Monitoring**: GPU usage, memory consumption
- **Model Artifacts**: Automatic model checkpointing
- **Experiment Comparison**: Compare different runs

### Log Files

Detailed logs are saved to `logs/training_YYYYMMDD_HHMMSS.log` with:
- Timestamped entries
- Debug information
- Error details
- Training progress

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `BATCH_SIZE` in `.env`
   - Use `DEVICE=cpu` for CPU-only training

2. **W&B Login Issues**:
   - Verify your API key is correct
   - Check internet connection
   - Ensure W&B project exists

3. **Data Loading Errors**:
   - Verify COCO format files exist
   - Check file paths in `.env`
   - Ensure proper permissions

### Error Analysis

The enhanced logging will help identify issues:

- **Invalid Loss Values**: NaN or infinite losses are detected and logged
- **Gradient Issues**: Gradient norm monitoring helps identify training instability
- **Batch Failures**: Detailed tracking of successful vs failed batches
- **Memory Issues**: GPU memory monitoring and warnings

## 📈 Understanding the Training Error

The original error you encountered (`loss.backward()` failure) was likely caused by:

1. **Loss Computation Issues**: The simplified loss function in `EfficientDetModel.compute_loss()` may produce invalid gradients
2. **Tensor Shape Mismatches**: Operations between tensors of incompatible shapes
3. **Memory Pressure**: GPU memory exhaustion during backpropagation
4. **Invalid Target Values**: Ground truth annotations with invalid values

The enhanced version includes:
- **Loss Validation**: Checks for NaN/infinite values before backpropagation
- **Gradient Monitoring**: Tracks gradient norms to detect instability
- **Error Recovery**: Continues training even if some batches fail
- **Detailed Logging**: Comprehensive error tracking and debugging information

## 🎯 Next Steps

1. **Review Training Plots**: Check the generated plots in `results/`
2. **Monitor W&B Dashboard**: Analyze training curves and metrics
3. **Evaluate Model**: Run inference on test set
4. **Hyperparameter Tuning**: Adjust learning rate, batch size, etc.
5. **Model Optimization**: Consider different architectures or loss functions

## 📝 Logging Levels

You can adjust logging verbosity by modifying the `setup_logging()` call in `main.py`:

```python
# For more detailed logging
logger, log_file = setup_logging(log_level=logging.DEBUG)

# For less verbose logging
logger, log_file = setup_logging(log_level=logging.WARNING)
```

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
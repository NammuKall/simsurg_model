# Model Factory Documentation

## Overview

The model factory system allows easy switching between different model architectures for training and inference. This implementation follows Method 1 (Factory Pattern with Configuration File) from the refactoring plan.

## Directory Structure

```
src/models/
├── __init__.py          # Exports all model classes
├── efficientdet.py      # EfficientDet-style model
├── resnet.py            # ResNet model
├── faster_rcnn.py       # Faster R-CNN model (torchvision)
├── model_config.py      # Model configuration definitions
├── model_factory.py     # Factory for creating models
└── README.md            # This file
```

## Available Models

### 1. EfficientDet
- **Name**: `EfficientDet`
- **Description**: EfficientDet-style model with ResNet50 backbone and feature fusion
- **Variants**:
  - `EfficientDet-Small`: Reduced anchors (5)
  - `EfficientDet-Large`: Increased anchors (12)
- **Default Parameters**: `num_classes=2`, `num_anchors=9`

### 2. ResNet
- **Name**: `ResNet`
- **Description**: ResNet model for classification
- **Variants**:
  - `ResNet-18`: Layers [2, 2, 2, 2]
  - `ResNet-34`: Layers [3, 4, 6, 3]
- **Default Parameters**: `num_classes=2`, `layers=[2, 2, 2, 2]`

### 3. FasterRCNN
- **Name**: `FasterRCNN`
- **Description**: Faster R-CNN with ResNet50-FPN backbone (torchvision implementation)
- **Variants**:
  - `FasterRCNN-Frozen`: Frozen backbone (0 trainable layers)
  - `FasterRCNN-Full`: Fully trainable backbone (5 trainable layers)
- **Default Parameters**: `num_classes=2`, `pretrained=True`, `trainable_backbone_layers=3`

## Usage

### Basic Usage

#### In Training Script (`main.py`)

The model is automatically created using the factory:

```python
from src.models.model_factory import create_model

# Model name is read from MODEL_NAME environment variable
# Default: EfficientDet
model = create_model(
    model_name=os.getenv("MODEL_NAME", "EfficientDet"),
    num_classes=2,
    device=device
)
```

#### In Inference Script (`inference.py`)

The model type is automatically detected from the checkpoint:

```python
from src.models.model_factory import create_model

# Model type is detected from checkpoint metadata
# Falls back to MODEL_NAME environment variable if not found
model = create_model(
    model_name=model_name,  # From checkpoint or env
    num_classes=2,
    device=device
)
```

### Configuration via Environment Variables

Add to your `.env` file:

```bash
# Model selection
MODEL_NAME=EfficientDet  # Options: EfficientDet, ResNet, FasterRCNN

# Optional: Model variant
MODEL_VARIANT=EfficientDet-Small  # Optional variant specification
```

### Programmatic Usage

```python
from src.models.model_factory import create_model, get_model_info, list_models

# List available models
available_models = list_models()
print(available_models)  # ['EfficientDet', 'ResNet', 'FasterRCNN']

# Get model information
info = get_model_info("FasterRCNN")
print(info['description'])
print(info['variants'])

# Create a model
model = create_model(
    model_name="FasterRCNN",
    num_classes=2,
    device=torch.device("cuda"),
    variant="FasterRCNN-Frozen"  # Optional
)
```

### Advanced Usage

#### Custom Parameters

You can override default parameters:

```python
model = create_model(
    model_name="EfficientDet",
    num_classes=2,
    num_anchors=12,  # Override default
    device=device
)
```

#### Model Variants

Use predefined variants:

```python
# Use EfficientDet-Small variant
model = create_model(
    model_name="EfficientDet",
    variant="EfficientDet-Small",
    num_classes=2,
    device=device
)
```

## Model Interface

All models follow a consistent interface:

### Forward Pass

```python
# Training mode (with targets)
outputs = model(images, targets)
# Returns: {'loss': tensor}

# Inference mode (without targets)
predictions = model(images)
# Returns: List of dicts with 'boxes', 'scores', 'labels'
```

### Target Format

All models expect targets in COCO format:

```python
targets = [
    {
        'boxes': torch.tensor([[x1, y1, x2, y2], ...]),  # Shape: [N, 4]
        'labels': torch.tensor([1, 2, ...]),              # Shape: [N], 1-indexed
        'image_id': torch.tensor([image_id])
    },
    # ... one dict per image in batch
]
```

### Prediction Format

All models return predictions in consistent format:

```python
predictions = [
    {
        'boxes': torch.tensor([[x1, y1, x2, y2], ...]),  # Shape: [M, 4]
        'scores': torch.tensor([0.9, 0.8, ...]),          # Shape: [M]
        'labels': torch.tensor([1, 2, ...])               # Shape: [M], 1-indexed
    },
    # ... one dict per image in batch
]
```

## Adding New Models

To add a new model:

1. **Create model file** (`src/models/new_model.py`):

```python
import torch.nn as nn

class NewModel(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        # Your model implementation
        pass
    
    def forward(self, x, targets=None):
        if targets is not None:
            # Training mode
            return {'loss': loss_tensor}
        else:
            # Inference mode
            return predictions_list
```

2. **Register in `__init__.py`**:

```python
from .new_model import NewModel

__all__ = [..., 'NewModel']
```

3. **Add to factory** (`model_factory.py`):

```python
from .new_model import NewModel

MODEL_REGISTRY = {
    # ... existing models
    "NewModel": NewModel,
}
```

4. **Add configuration** (`model_config.py`):

```python
MODEL_CONFIGS = {
    # ... existing configs
    "NewModel": ModelConfig(
        name="NewModel",
        class_name="NewModel",
        default_params={"num_classes": 2},
        description="Description of your model",
        supports_pretrained=True,
    ),
}
```

## Model Checkpoint Format

Model checkpoints now include metadata:

```python
checkpoint = {
    'model_state_dict': {...},
    'best_model_state': {...},
    'optimizer_state_dict': {...},
    'model_architecture': 'FasterRCNN',  # Model name
    'model_variant': 'FasterRCNN-Frozen',  # Variant (if used)
    'num_classes': 2,
    # ... other training metadata
}
```

This allows automatic model type detection during inference.

## Examples

### Switch Models for Training

```bash
# Train with EfficientDet
MODEL_NAME=EfficientDet python main.py

# Train with Faster R-CNN
MODEL_NAME=FasterRCNN python main.py

# Train with Faster R-CNN (frozen backbone)
MODEL_NAME=FasterRCNN MODEL_VARIANT=FasterRCNN-Frozen python main.py
```

### Run Inference

```bash
# Inference automatically detects model type from checkpoint
python inference.py --model models/model_20240101_120000.pth

# Or specify model type explicitly
MODEL_NAME=FasterRCNN python inference.py --model models/model_20240101_120000.pth
```

## Troubleshooting

### Model Not Found Error

If you see `ValueError: Unknown model: 'ModelName'`:

1. Check that the model is registered in `MODEL_REGISTRY`
2. Verify the model name matches exactly (case-sensitive)
3. Use `list_models()` to see available models

### Checkpoint Loading Issues

If inference fails to load a checkpoint:

1. Check that the checkpoint contains `model_architecture` key
2. Set `MODEL_NAME` environment variable explicitly
3. Verify the model architecture matches the checkpoint

### Faster R-CNN Specific Notes

- Faster R-CNN uses torchvision's implementation
- Requires `torchvision >= 0.15.0`
- Labels are automatically converted between 0-indexed (torchvision) and 1-indexed (our format)
- Background class (label 0) is filtered out in inference

## Performance Comparison

When comparing models:

1. **EfficientDet**: Good balance of speed and accuracy, custom implementation
2. **FasterRCNN**: High accuracy, well-tested implementation, slower training
3. **ResNet**: Classification model, not optimized for detection

Use W&B to track and compare model performance across different architectures.


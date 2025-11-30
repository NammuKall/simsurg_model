# Model Refactoring Plan

## Current State

Currently, model creation is hardcoded in two places:
1. **`main.py`** (line 248): `model = EfficientDetModel(num_classes=2).to(device)`
2. **`inference.py`** (line 246): `model = EfficientDetModel(num_classes=2).to(device)`

Both files directly import and instantiate `EfficientDetModel` from `src.models`.

## Goal

Refactor model creation into a separate module that allows easy switching between different model architectures while maintaining backward compatibility.

---

## Method 1: Factory Pattern with Configuration File (Recommended)

### Overview
Create a model factory that reads model configuration from environment variables or a config file, allowing easy switching via configuration.

### Implementation Structure

```
src/
  ├── model_factory.py          # New: Factory for creating models
  ├── model_config.py            # New: Model configuration definitions
  └── models.py                  # Existing: Model definitions
```

### Pros
- ✅ Clean separation of concerns
- ✅ Easy to add new models
- ✅ Configuration-driven (no code changes needed)
- ✅ Type-safe with proper error handling
- ✅ Supports model-specific hyperparameters

### Cons
- ⚠️ Requires configuration management
- ⚠️ Slightly more complex initial setup

### Example Usage
```python
# In main.py
from src.model_factory import create_model

model = create_model(
    model_name=os.getenv("MODEL_NAME", "EfficientDet"),
    num_classes=2,
    device=device,
    **model_kwargs
)
```

### Configuration Options
- Environment variable: `MODEL_NAME=EfficientDet`
- Config file: `config.yaml` with model settings
- Command-line argument support

---

## Method 2: Registry Pattern with Decorators

### Overview
Use a decorator-based registry system where models register themselves. Models can be selected by name at runtime.

### Implementation Structure

```
src/
  ├── model_registry.py          # New: Registry for model classes
  └── models.py                  # Modified: Models register themselves
```

### Pros
- ✅ Very Pythonic and elegant
- ✅ Self-documenting (models register themselves)
- ✅ Easy to discover available models
- ✅ No central factory needed

### Cons
- ⚠️ Requires importing all model modules to register them
- ⚠️ Less explicit than factory pattern

### Example Usage
```python
# In main.py
from src.model_registry import get_model

model_class = get_model(os.getenv("MODEL_NAME", "EfficientDet"))
model = model_class(num_classes=2).to(device)
```

### Example Registration
```python
# In models.py
from src.model_registry import register_model

@register_model("EfficientDet")
class EfficientDetModel(nn.Module):
    ...
```

---

## Method 3: Strategy Pattern with Abstract Base Class

### Overview
Define an abstract base class for all models and use a strategy pattern to switch between implementations. Each model implements a common interface.

### Implementation Structure

```
src/
  ├── base_model.py             # New: Abstract base class
  ├── model_factory.py           # New: Factory using strategy pattern
  └── models/
      ├── __init__.py
      ├── efficientdet.py        # Moved from models.py
      ├── resnet.py              # Moved from models.py
      └── yolo.py                # Example: Future model
```

### Pros
- ✅ Enforces consistent interface
- ✅ Type checking support
- ✅ Clear model hierarchy
- ✅ Easy to add new models following the pattern

### Cons
- ⚠️ More boilerplate code
- ⚠️ Requires refactoring existing models

### Example Usage
```python
# In main.py
from src.model_factory import ModelFactory

factory = ModelFactory()
model = factory.create(
    model_type="EfficientDet",
    num_classes=2,
    device=device
)
```

---

## Method 4: Simple Function-Based Factory (Simplest)

### Overview
Create a simple factory function that uses a dictionary mapping model names to model classes. Minimal changes to existing code.

### Implementation Structure

```
src/
  └── model_factory.py          # New: Simple factory function
```

### Pros
- ✅ Simplest to implement
- ✅ Minimal code changes
- ✅ Easy to understand
- ✅ Quick to set up

### Cons
- ⚠️ Less flexible than other methods
- ⚠️ No automatic discovery
- ⚠️ Manual registration required

### Example Usage
```python
# In main.py
from src.model_factory import create_model

model = create_model(
    model_name=os.getenv("MODEL_NAME", "EfficientDet"),
    num_classes=2,
    device=device
)
```

### Example Implementation
```python
# model_factory.py
from src.models import EfficientDetModel, ResNet

MODEL_REGISTRY = {
    "EfficientDet": EfficientDetModel,
    "ResNet": ResNet,
}

def create_model(model_name: str, num_classes: int = 2, device=None, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(num_classes=num_classes, **kwargs)
    
    if device:
        model = model.to(device)
    
    return model
```

---

## Method 5: Configuration-Based with YAML/JSON

### Overview
Store model configurations in YAML or JSON files. The factory reads configurations and instantiates models accordingly. Supports model variants and hyperparameters.

### Implementation Structure

```
configs/
  ├── models.yaml               # Model configurations
src/
  └── model_factory.py          # New: Factory that reads configs
```

### Pros
- ✅ Very flexible
- ✅ Easy to version control configurations
- ✅ Supports model variants and hyperparameters
- ✅ Can define multiple model configurations

### Cons
- ⚠️ Requires YAML/JSON parsing
- ⚠️ More complex than simple factory
- ⚠️ Configuration validation needed

### Example Configuration
```yaml
# configs/models.yaml
models:
  EfficientDet:
    class: EfficientDetModel
    default_params:
      num_classes: 2
      num_anchors: 9
    variants:
      EfficientDet-Small:
        num_anchors: 5
      EfficientDet-Large:
        num_anchors: 12
  
  ResNet:
    class: ResNet
    default_params:
      num_classes: 2
```

---

## Comparison Matrix

| Method | Complexity | Flexibility | Maintainability | Setup Time | Recommended For |
|--------|-----------|-------------|-----------------|------------|-----------------|
| **Method 1: Factory + Config** | Medium | High | High | Medium | Production systems |
| **Method 2: Registry Pattern** | Medium | High | High | Medium | Large codebases |
| **Method 3: Strategy + ABC** | High | High | Very High | High | Enterprise projects |
| **Method 4: Simple Factory** | Low | Medium | Medium | Low | Quick refactoring |
| **Method 5: YAML/JSON Config** | Medium | Very High | High | Medium | Research/experiments |

---

## Recommended Approach: Hybrid (Method 1 + Method 4)

### Why Hybrid?
- Start with **Method 4** (Simple Factory) for immediate refactoring
- Evolve to **Method 1** (Factory + Config) as needs grow
- Provides immediate benefits with room for expansion

### Implementation Steps

#### Phase 1: Quick Refactoring (Method 4)
1. Create `src/model_factory.py` with simple dictionary-based registry
2. Update `main.py` to use factory
3. Update `inference.py` to use factory
4. Add environment variable support (`MODEL_NAME`)

#### Phase 2: Enhanced Configuration (Method 1)
1. Add configuration file support
2. Add model-specific parameter handling
3. Add validation and error handling
4. Add model discovery utilities

---

## Detailed Implementation Plan for Recommended Approach

### Step 1: Create Model Factory (`src/model_factory.py`)

```python
"""
Model factory for creating and managing different model architectures.
Supports easy switching between models via configuration.
"""
import os
import torch
import logging
from typing import Optional, Dict, Any
from src.models import EfficientDetModel, ResNet

logger = logging.getLogger(__name__)

# Model registry mapping names to classes
MODEL_REGISTRY: Dict[str, type] = {
    "EfficientDet": EfficientDetModel,
    "ResNet": ResNet,
    # Add more models here as they're implemented
}

def list_available_models() -> list:
    """Return list of available model names."""
    return list(MODEL_REGISTRY.keys())

def create_model(
    model_name: Optional[str] = None,
    num_classes: int = 2,
    device: Optional[torch.device] = None,
    **model_kwargs
) -> torch.nn.Module:
    """
    Create a model instance.
    
    Args:
        model_name: Name of the model to create. If None, reads from MODEL_NAME env var.
        num_classes: Number of classes for the model
        device: Device to move model to (optional)
        **model_kwargs: Additional model-specific arguments
    
    Returns:
        Initialized model instance
    
    Raises:
        ValueError: If model_name is not in registry
    """
    # Get model name from parameter or environment
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", "EfficientDet")
    
    model_name = model_name.strip()
    
    # Validate model name
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(list_available_models())
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available models: {available}"
        )
    
    # Get model class
    model_class = MODEL_REGISTRY[model_name]
    
    # Create model instance
    logger.info(f"Creating {model_name} model with {num_classes} classes")
    try:
        model = model_class(num_classes=num_classes, **model_kwargs)
    except Exception as e:
        logger.error(f"Failed to create {model_name} model: {e}")
        raise
    
    # Move to device if specified
    if device is not None:
        model = model.to(device)
        logger.info(f"Model moved to device: {device}")
    
    return model

def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a model."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_class = MODEL_REGISTRY[model_name]
    return {
        "name": model_name,
        "class": model_class.__name__,
        "module": model_class.__module__,
        "doc": model_class.__doc__,
    }
```

### Step 2: Update `main.py`

**Before:**
```python
from src.models import EfficientDetModel
...
model = EfficientDetModel(num_classes=2).to(device)
```

**After:**
```python
from src.model_factory import create_model
...
model = create_model(
    model_name=os.getenv("MODEL_NAME", "EfficientDet"),
    num_classes=2,
    device=device
)
```

### Step 3: Update `inference.py`

**Before:**
```python
from src.models import EfficientDetModel
...
model = EfficientDetModel(num_classes=2).to(device)
```

**After:**
```python
from src.model_factory import create_model
...
model = create_model(
    model_name=os.getenv("MODEL_NAME", "EfficientDet"),
    num_classes=2,
    device=device
)
```

### Step 4: Update Environment Configuration

Add to `.env` or `env.example`:
```bash
# Model selection
MODEL_NAME=EfficientDet  # Options: EfficientDet, ResNet
```

### Step 5: Add Model-Specific Configuration (Optional Enhancement)

```python
# In model_factory.py
MODEL_CONFIGS = {
    "EfficientDet": {
        "default_num_anchors": 9,
        "supports_pretrained": True,
    },
    "ResNet": {
        "default_layers": [2, 2, 2, 2],
        "supports_pretrained": True,
    },
}
```

---

## Migration Checklist

- [ ] Create `src/model_factory.py`
- [ ] Update `main.py` to use factory
- [ ] Update `inference.py` to use factory
- [ ] Update `env.example` with `MODEL_NAME`
- [ ] Add unit tests for model factory
- [ ] Update documentation/README
- [ ] Test with existing EfficientDet model
- [ ] Test model switching functionality
- [ ] Verify backward compatibility

---

## Future Enhancements

1. **Model Loading from Checkpoints**: Add utility to detect model type from checkpoint
2. **Model Comparison**: Add utilities to compare different models
3. **Model Profiling**: Add utilities to profile model performance
4. **Auto Model Selection**: Add logic to automatically select best model based on dataset
5. **Model Versioning**: Add support for model versioning and compatibility checking

---

## Questions to Consider

1. **Do you need to support loading different model architectures from the same checkpoint?**
   - If yes, add checkpoint metadata to store model type
   
2. **Do you need model-specific hyperparameters?**
   - If yes, use Method 1 or Method 5
   
3. **How many models do you plan to support?**
   - Few (<5): Method 4 is sufficient
   - Many (>5): Consider Method 1 or Method 2
   
4. **Do you need dynamic model discovery?**
   - If yes, use Method 2 (Registry Pattern)

---

## Recommendation

**Start with Method 4 (Simple Factory)** for immediate refactoring, then evolve to **Method 1 (Factory + Config)** as your needs grow. This provides:
- ✅ Quick implementation
- ✅ Immediate benefits
- ✅ Room for future expansion
- ✅ Minimal breaking changes


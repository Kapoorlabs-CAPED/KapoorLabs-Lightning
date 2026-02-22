# Lightning Modules Architecture

## Overview

This package contains a hierarchical structure of PyTorch Lightning modules designed for clean separation of concerns and code reusability.

## Module Hierarchy

```
BaseModule (base_module.py)
    ├── ClassificationModule (classification_module.py)
    └── AutoEncoderModule (autoencoder_module.py)
```

## BaseModule

**Location:** `base_module.py`

**Purpose:** Foundational class containing common functionality for all Lightning modules.

**Key Features:**
- Optimizer configuration
- Learning rate scheduling
- Logging utilities with `log_metrics()`
- Checkpoint loading with `load_checkpoint()`
- Pretrained weight loading with `load_pretrained()`
- JSON configuration parsing with `extract_json()`
- Automatic learning rate logging at epoch end

**Core Methods:**
- `forward()`: Network forward pass
- `loss()`: Loss computation
- `configure_optimizers()`: Optimizer and scheduler setup
- `on_train_epoch_end()`: End-of-epoch actions (LR logging, scheduler step)
- `log_metrics()`: Unified logging interface

**Class Method:**
- `load_checkpoint()`: Load model from checkpoint file (latest or specified path)
- `extract_json()`: Parse JSON configuration files

## ClassificationModule

**Location:** `classification_module.py`

**Inherits:** `BaseModule`

**Purpose:** Specialized module for classification tasks including multi-task learning.

**Key Features:**
- Standard multi-class classification
- ONEAT multi-task accuracy (class, xyz coordinates, dimensions, confidence)
- Automatic accuracy computation and logging
- Support for both simple and complex accuracy metrics

**Core Methods:**
- `training_step()`: Training loop with loss and accuracy logging
- `validation_step()`: Validation evaluation
- `test_step()`: Test evaluation
- `compute_accuracy()`: Flexible accuracy computation (standard or ONEAT)
- `_shared_eval()`: Shared evaluation logic for val/test
- `_log_accuracy()`: Unified accuracy logging

**Parameters:**
- `num_classes`: Number of output classes
- `oneat_accuracy`: Enable multi-task accuracy computation

## AutoEncoderModule

**Location:** `autoencoder_module.py`

**Inherits:** `BaseModule`

**Purpose:** Specialized module for autoencoder training and inference.

**Key Features:**
- Autoencoder-specific training (reconstruction loss)
- Scaled prediction for point cloud applications
- Support for latent feature extraction

**Core Methods:**
- `training_step()`: Reconstruction training
- `predict_step()`: Scaled prediction with mean restoration
- `_shared_eval()`: Autoencoder evaluation logic

**Parameters:**
- `scale_z`: Z-axis scaling factor for predictions
- `scale_xy`: XY-axis scaling factor for predictions

## Usage Examples

### Classification Module

```python
from kapoorlabs_lightning import ClassificationModule
from kapoorlabs_lightning.pytorch_models import DenseNet3D

network = DenseNet3D(input_channels=1, num_classes=3)
loss_func = torch.nn.CrossEntropyLoss()
optim_func = lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9)

model = ClassificationModule(
    network=network,
    loss_func=loss_func,
    optim_func=optim_func,
    num_classes=3,
    oneat_accuracy=False
)
```

### AutoEncoder Module

```python
from kapoorlabs_lightning import AutoEncoderModule

network = CloudAutoEncoder(...)
loss_func = torch.nn.MSELoss()
optim_func = lambda params: torch.optim.Adam(params, lr=0.001)

model = AutoEncoderModule(
    network=network,
    loss_func=loss_func,
    optim_func=optim_func,
    scale_z=1.0,
    scale_xy=1.0
)
```

### Loading from Checkpoint

```python
from kapoorlabs_lightning import ClassificationModule

model, network = ClassificationModule.load_checkpoint(
    network=my_network,
    loss_func=my_loss,
    optim_func=my_optimizer,
    checkpoint_dir="/path/to/checkpoints/",  # Loads latest .ckpt
)

model, network = ClassificationModule.load_checkpoint(
    network=my_network,
    loss_func=my_loss,
    optim_func=my_optimizer,
    checkpoint_path="/path/to/specific_checkpoint.ckpt",  # Loads specific file
)
```

## Backward Compatibility

Legacy class names are aliased for compatibility:
- `LightningModel` → `ClassificationModule`
- `AutoLightningModel` → `AutoEncoderModule`

## Design Principles

1. **Separation of Concerns**: Each module handles a specific task type
2. **DRY (Don't Repeat Yourself)**: Common functionality in BaseModule
3. **Clean Interfaces**: Consistent method signatures across modules
4. **Extensibility**: Easy to add new module types by inheriting BaseModule
5. **Well-Engineered**: Proper logging, error handling, and documentation

## Adding New Modules

To add a new specialized module:

1. Create new file in `lightning_modules/`
2. Inherit from `BaseModule`
3. Override task-specific methods (`training_step`, `validation_step`, etc.)
4. Add to `__init__.py` exports
5. Document in this README

Example:

```python
from .base_module import BaseModule

class MyCustomModule(BaseModule):
    def __init__(self, network, loss_func, optim_func, **kwargs):
        super().__init__(network, loss_func, optim_func, **kwargs)

    def training_step(self, batch, batch_idx):
        # Custom training logic
        pass
```

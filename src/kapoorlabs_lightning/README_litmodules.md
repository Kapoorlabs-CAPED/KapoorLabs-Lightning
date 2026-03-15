# Lightning Modules Architecture

## Overview

This package contains a hierarchical structure of PyTorch Lightning modules designed for clean separation of concerns and code reusability.

## Module Hierarchy

```
BaseModule (base_module.py)
    ├── OneatActionModule (oneat_module.py)
    ├── CellFateModule (cellfate_module.py)
    └── CareModule (care_module.py)
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

## OneatActionModule

**Location:** `oneat_module.py`

**Inherits:** `BaseModule`

**Purpose:** Specialized module for ONEAT spatio-temporal event detection with YOLO-style multi-task learning.

**Key Features:**
- Multi-task loss logging (classification, localization, confidence)
- ONEAT-specific accuracy metrics (class, xyzt, hwd, confidence)
- Prediction step for cell-wise event classification
- Handles different tensor formats for model output vs ground truth

**Core Methods:**
- `training_step()`: Training with multi-component loss logging
- `validation_step()`: Validation evaluation
- `test_step()`: Test evaluation
- `compute_accuracy()`: Multi-task accuracy computation
- `predict_step()`: Cell-wise prediction for inference

**Tensor Formats:**
```
Ground Truth (YOLO label): [x, y, z, t, h, w, d, c] + [one-hot categories]
Model Output:              [categories (softmax)] + [x, y, z, t, h, w, d, c (sigmoid)]
```

**Parameters:**
- `num_classes`: Number of event classes
- `oneat_accuracy`: Enable multi-task accuracy (class + box metrics)
- `imagex, imagey, imagez`: Patch dimensions
- `size_tminus, size_tplus`: Temporal window parameters
- `event_names`: List of event class names

**Usage:**
```python
from kapoorlabs_lightning import OneatActionModule
from kapoorlabs_lightning.pytorch_losses import VolumeYoloLoss

network = DenseVollNet(input_shape=(3, 8, 64, 64), categories=2, box_vector=8)
loss_func = VolumeYoloLoss(categories=2, box_vector=8, device="cuda", return_components=True)
optim_func = lambda params: torch.optim.Adam(params, lr=1e-3)

model = OneatActionModule(
    network=network,
    loss_func=loss_func,
    optim_func=optim_func,
    num_classes=2,
    oneat_accuracy=True,
)
```

## Loss Functions

### VolumeYoloLoss

**Location:** `pytorch_losses.py`

**Purpose:** Multi-task loss for YOLO-style volumetric event detection.

**Components:**
1. **Classification Loss**: NLLLoss (model outputs softmax probabilities)
2. **Localization Loss**: MSE for xyzt position + sqrt-MSE for hwd dimensions
3. **Confidence Loss**: MSE for objectness score

**Key Feature:** Handles different tensor orderings between model output and ground truth.

```python
from kapoorlabs_lightning.pytorch_losses import VolumeYoloLoss

loss_func = VolumeYoloLoss(
    categories=2,
    box_vector=8,
    device="cuda",
    class_weights_dict={0: 1.0, 1: 5.0},  # Optional class weighting
    return_components=True,  # Return individual loss components
)

# Forward pass
combined_loss, loss_xyzt_hwd, loss_conf, loss_class = loss_func(predictions, targets)
```

### OneatClassificationLoss

**Location:** `pytorch_losses.py`

**Purpose:** Simple classification loss when box regression is not needed.

```python
from kapoorlabs_lightning.pytorch_losses import OneatClassificationLoss

loss_func = OneatClassificationLoss(
    categories=2,
    class_weights_dict={0: 1.0, 1: 3.0},
)
```

## Dataset Classes

### H5VisionDataset (Training)

**Location:** `pytorch_datasets.py`

**Purpose:** Memory-efficient streaming of pre-extracted patches from HDF5 files.

```python
from kapoorlabs_lightning import H5VisionDataset

dataset = H5VisionDataset(
    h5_file="training_data.h5",
    split="train",                    # "train" or "val"
    transforms=train_transforms,
    return_segmentation=False,
    num_classes=2,
    compute_class_weights=True,
)

# Returns: (image, label) where label is YOLO format
# image: (T, Z, Y, X) float32
# label: (8 + num_classes,) float32 - [box_vector, one_hot_class]
```

**Features:**
- SWMR mode for safe concurrent reads
- Automatic class weight computation from YOLO labels
- Lazy loading (only loads requested samples)

### OneatPredictionDataset (Inference)

**Location:** `oneat_prediction_dataset.py`

**Purpose:** Yields temporal windows from full timelapse for cell-wise prediction.

```python
from kapoorlabs_lightning import OneatPredictionDataset

dataset = OneatPredictionDataset(
    raw_file="timelapse.tif",
    seg_file="segmentation.tif",
    size_tminus=1,
    size_tplus=1,
    normalize=True,
    pmin=1.0,
    pmax=99.8,
)

# Returns: (temporal_raw, temporal_seg, timepoint, metadata)
```

**Prediction Flow:**
1. Dataset yields temporal windows (raw + seg) for each timepoint
2. `predict_step` finds all cells in segmentation
3. For each cell: extracts patch from raw image around cell centroid
4. Classifies patch and records global coordinates for positive events

---

## Usage Examples

### ONEAT Training with MitosisInception

```python
from kapoorlabs_lightning import MitosisInception

trainer = MitosisInception(
    h5_file="training_data.h5",
    num_classes=2,
    epochs=100,
    batch_size=32,
    learning_rate=1e-3,
)

# Setup model
trainer.setup_densenet_vision_model(
    input_shape=(3, 8, 64, 64),
    categories=2,
    box_vector=8,
)

# Setup transforms (light/medium/heavy)
trainer.setup_oneat_transforms_medium()

# Setup dataset with class weighting
trainer.setup_vision_h5_datasets(compute_class_weights=True)

# Setup optimizer (Adam recommended for ONEAT)
trainer.setup_adam()

# Setup scheduler
trainer.setup_learning_rate_scheduler()

# Create Lightning module
trainer.setup_oneat_lightning_model()

# Train
trainer.train(logger=logger, callbacks=callbacks)
```

### Loading from Checkpoint

```python
from kapoorlabs_lightning import OneatActionModule

model, network = OneatActionModule.load_checkpoint(
    network=my_network,
    loss_func=my_loss,
    optim_func=my_optimizer,
    checkpoint_dir="/path/to/checkpoints/",  # Loads latest .ckpt
)

model, network = OneatActionModule.load_checkpoint(
    network=my_network,
    loss_func=my_loss,
    optim_func=my_optimizer,
    checkpoint_path="/path/to/specific_checkpoint.ckpt",  # Loads specific file
)
```

## CareModule

**Location:** `care_module.py`

**Inherits:** `BaseModule`

**Purpose:** Supervised 3D denoising using paired low/high SNR volumes. Uses MSE loss and logs PSNR on validation. Supports tiled prediction with linear-blend overlap stitching for large volumes.

**Key Features:**
- MSE training loss
- PSNR logging on validation (`val_psnr`)
- `predict_step` tiles a full 3D volume, runs batched inference, and stitches with linear blend weights
- `stitch_tiles()` helper exported at package level

**Core Methods:**
- `training_step()`: MSE loss on low→high pairs
- `validation_step()`: MSE + PSNR logging
- `predict_step()`: Tiled prediction for large 3D volumes

**Parameters:**
- `n_tiles`: Number of tiles per axis, e.g. `[1, 4, 4]`
- `tile_overlap`: Fractional overlap between tiles (default `0.125`)
- `eval_transforms`: Percentile normalization applied at inference time

**Usage:**
```python
from kapoorlabs_lightning import CareModule
import torch

network = UNet(conv_dims=3, in_channels=1, num_classes=1, depth=3)
loss_func = torch.nn.MSELoss()
optim_func = torch.optim.Adam

model = CareModule(
    network=network,
    loss_func=loss_func,
    optim_func=optim_func,
    n_tiles=[1, 4, 4],
    tile_overlap=0.125,
    eval_transforms=eval_preset,
)
```

---

## Design Principles

1. **Separation of Concerns**: Each module handles a specific task type
2. **DRY (Don't Repeat Yourself)**: Common functionality in BaseModule
3. **Clean Interfaces**: Consistent method signatures across modules
4. **Extensibility**: Easy to add new module types by inheriting BaseModule
5. **Format Handling**: Proper handling of different tensor orderings (model vs GT)

## Adding New Modules

To add a new specialized module:

1. Create new file in the package directory
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

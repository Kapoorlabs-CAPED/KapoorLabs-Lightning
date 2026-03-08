# Cell Fate Classification: Time Series Models for Tracking Data

Classify cell fates (basal, goblet, radial, etc.) from 25-timepoint tracking trajectories using 1D DenseNet with attention pooling.

## Installation

```bash
pip install kapoorlabs-lightning
```

## Workflow Overview

```
1. Generate H5 Dataset    →    2. Train Model    →    3. Predict Cell Fates
   (from tracking data)         (InceptionNet)        (val accuracy)
```

---

## 1. Training Data Format

The H5 file contains time series extracted from cell tracking data.

**H5 Structure:**
```
cellfate_data.h5
├── train_arrays    (N, 25, num_features) float32
├── train_labels    (N,) int64
├── val_arrays      (M, 25, num_features) float32
└── val_labels      (M,) int64
```

**Features per timepoint (typical):**
- Shape: Radius, Eccentricity (3 components), Local_Cell_Density, Surface_Area
- Dynamic: Speed, Motion_Angle (Z/Y/X), Acceleration, Distance_Cell_mask, Radial_Angle (Z/Y/X), Cell_Axis (Z/Y/X)
- Total: ~18 features per timepoint

**Labels:**
- Integer class indices (e.g., 0=basal, 1=goblet, 2=radial)

---

## 2. Train Model

```bash
cd scripts/model_training
python lightning-cellfate.py
```

### Configuration

```bash
# Different transform presets
python lightning-cellfate.py parameters.transform_preset=light
python lightning-cellfate.py parameters.transform_preset=heavy

# Different model architectures
python lightning-cellfate.py parameters.model_choice=inception   # default
python lightning-cellfate.py parameters.model_choice=densenet
python lightning-cellfate.py parameters.model_choice=mitosisnet
```

**Config:** `scripts/conf/scenario_train_cellfate.yaml`

---

## 3. Evaluate

```bash
cd scripts/model_prediction
python compute_cellfate_accuracy.py
```

Outputs per-class accuracy, precision, and overall accuracy as a pandas DataFrame and CSV.

---

## Model Architectures

### InceptionNet (Recommended)

1D DenseNet with attention-based pooling for time series classification.

```
Input: (batch, channels, 25)  # channels = num_features, 25 = timepoints
         │
    ┌────▼────┐
    │ Conv1d  │  kernel=7, features → num_init_features
    │  Stem   │
    └────┬────┘
         │
    ┌────▼────┐
    │  Dense  │  Configurable depth per block
    │  Blocks │  (default: 6, 12, 24, 16)
    └────┬────┘
         │
    ┌────▼────┐
    │Attention│  Multi-head attention with [CLS] token
    │  Pool   │  (8 heads, learnable positional embeddings)
    └────┬────┘
         │
    ┌────▼────┐
    │ Linear  │  → num_classes
    └─────────┘
```

### DenseNet

Pure 1D DenseNet with adaptive average pooling (no attention).

### MitosisNet

Simple baseline: Conv1d → MaxPool → Conv1d → MaxPool → FC.

---

## Transform Presets

All cell fate transforms preserve temporal order. No time shifts, permutations, or warping.

### Light
```yaml
- Gaussian noise (std=0.01)
- Random scaling (0.98 - 1.02)
```

### Medium
```yaml
- Gaussian noise (std=0.02)
- Random scaling (0.95 - 1.05)
- Random masking (up to 10% of timepoints)
```

### Heavy
```yaml
- Gaussian noise (std=0.03)
- Random scaling (0.9 - 1.1)
- Random masking (up to 20% of timepoints)
```

---

## Configuration

### Model Parameters
```yaml
# cellfate.yaml
growth_rate: 32
block_config: [6, 12, 24, 16]
num_init_features: 32
bottleneck_size: 4
kernel_size: 7
attn_heads: 8
seq_len: 25
num_classes: 3
model_choice: 'inception'
```

### Training Parameters
```yaml
learning_rate: 0.001
batch_size: 64
epochs: 250
transform_preset: 'medium'
```

### Data Paths
```yaml
# cellfate_default.yaml
base_data_dir: '/path/to/data/'
cellfate_h5_file: 'cellfate_data.h5'
log_path: '/path/to/model/'
experiment_name: 'cellfate_default'
```

---

## Directory Structure

```
scripts/
├── conf/
│   ├── parameters/cellfate.yaml
│   ├── train_data_paths/cellfate_default.yaml
│   ├── scenario_train_cellfate.yaml
│   └── scenario_predict_cellfate.yaml
├── model_training/
│   ├── lightning-cellfate.py
│   └── scenario_train_cellfate.py
└── model_prediction/
    ├── compute_cellfate_accuracy.py
    └── scenario_predict_cellfate.py

src/kapoorlabs_lightning/
├── cellfate_module.py         # CellFateModule (BaseModule subclass)
├── time_series_presets.py     # Order-preserving transform presets
├── pytorch_models.py          # InceptionNet, DenseNet, MitosisNet
└── pytorch_datasets.py        # H5MitosisDataset
```

---

## License

BSD-3-Clause

## Citation

If using this software, cite KapoorLabs-Lightning.

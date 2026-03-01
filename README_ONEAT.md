# ONEAT: Spatio-Temporal Event Detection for 3D+Time Microscopy

Action classification for TZYX microscopy timelapses using DenseVollNet architecture with YOLO-style multi-task learning.

## Installation

```bash
pip install kapoorlabs-lightning
```

## Workflow Overview

```
1. Generate H5 Dataset    →    2. Train Model    →    3. Predict Events
   (with YOLO labels)          (Adam optimizer)       (NMS filtering)
```

---

## 1. Generate Training Data

```bash
cd scripts/train_data_generation
python generate-oneat-training-data.py
```

**Inputs:**
- Raw TZYX timelapse images (`*.tif`)
- Segmentation masks with cell labels
- CSV files with event annotations (`t,z,y,x,class`)

**Process:**
1. Normalizes raw images in chunks (50 timepoints)
2. Extracts temporal patches around annotated events
3. Computes YOLO-style bounding box labels from segmentation
4. Creates train/val split (default 80/20)
5. Saves to HDF5 file with streaming architecture

**Output H5 Structure:**
```
training_data.h5
├── train/
│   ├── images      (N, T, Z, Y, X) float32
│   ├── seg         (N, T, Z, Y, X) uint16
│   └── labels      (N, 8 + num_classes) float32  # YOLO format
└── val/
    ├── images
    ├── seg
    └── labels
```

**YOLO Label Format:**
```
[x, y, z, t, h, w, d, c] + [one-hot class]
 │  │  │  │  │  │  │  │     └── Event class (e.g., [1,0] = normal, [0,1] = mitosis)
 │  │  │  │  │  │  │  └── Confidence (always 1.0 for GT)
 │  │  │  │  │  │  └── Depth (z-extent / crop_z)
 │  │  │  │  │  └── Width (x-extent / crop_x)
 │  │  │  │  └── Height (y-extent / crop_y)
 │  │  │  └── Time position (always 0.5 = center)
 │  │  └── Z position (always 0.5 = center)
 │  └── Y position (always 0.5 = center)
 └── X position (always 0.5 = center)
```

**Config:** `scripts/conf/scenario_generate_oneat.yaml`

---

## 2. Train Model

### With Adam Optimizer (Recommended)

```bash
cd scripts/model_training
python lightning-oneat-adam.py
```

### With SGD Optimizer

```bash
python lightning-oneat.py
```

### SLURM Submission

```bash
# Single job
sbatch slurm_train_adam_heavy.sh

# Learning rate sweep (parallel jobs)
python submit_lr_sweep_adam_heavy.py
```

**Training Steps:**
1. Instantiate DenseVollNet model
2. Setup transforms (light/medium/heavy)
3. Load H5 dataset with YOLO labels
4. Configure VolumeYoloLoss (multi-task)
5. Configure optimizer (Adam recommended)
6. Configure scheduler (Cosine/WarmCosine)
7. Train with Lightning

**Config:** `scripts/conf/scenario_train_oneat.yaml`

---

## 3. Predict

```bash
cd scripts/model_prediction
python lightning-predict-oneat.py
```

**Process:**
1. Load model checkpoint
2. Read raw + seg timelapse pairs
3. Normalize raw in chunks (same as training)
4. For each timepoint:
   - Extract patches around segmented cells
   - Classify action (argmax, no threshold)
5. Apply NMS in space-time
6. Save CSV per event type

**Output:**
```csv
t,z,y,x
10,5,128,256
15,6,130,258
```

**Config:** `scripts/conf/scenario_predict_oneat.yaml`

---

## Dataset Classes

### Training: H5VisionDataset

Streams pre-extracted patches from HDF5 files for memory-efficient training.

```python
from kapoorlabs_lightning import H5VisionDataset

dataset = H5VisionDataset(
    h5_file="training_data.h5",
    split="train",                    # or "val"
    transforms=train_transforms,
    return_segmentation=False,
    num_classes=2,
    compute_class_weights=True,       # Handle class imbalance
)

# Access class weights for loss function
class_weights = dataset.get_class_weights()  # {0: 1.0, 1: 4.5}
```

**Features:**
- Lazy loading from H5 (low memory footprint)
- SWMR mode for safe concurrent reads
- Automatic class weight computation from YOLO labels
- Optional segmentation return for visualization

**H5 Structure:**
```
train/
  ├── images  (N, T, Z, Y, X)  # Normalized patches
  ├── labels  (N, 8 + C)       # YOLO labels [box_vector + one-hot]
  └── seg     (N, T, Z, Y, X)  # Optional segmentation patches
```

---

### Prediction: OneatPredictionDataset

Loads full timelapse images and yields temporal windows for cell-by-cell prediction.

```python
from kapoorlabs_lightning import OneatPredictionDataset

dataset = OneatPredictionDataset(
    raw_file="timelapse_raw.tif",     # Full TZYX image
    seg_file="timelapse_seg.tif",     # Instance segmentation
    size_tminus=1,
    size_tplus=1,
    normalize=True,
    pmin=1.0,
    pmax=99.8,
    chunk_steps=50,                   # Normalization chunk size
)
```

**Prediction Workflow:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    OneatPredictionDataset                       │
│  Loads: raw_image (T,Z,Y,X) + seg_image (T,Z,Y,X)              │
│  Yields: temporal_window (t-1:t+2) for each valid timepoint    │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OneatActionModule.predict_step               │
│                                                                 │
│  1. Find all cell instances in seg at timepoint t              │
│     └── cell_ids = unique(seg[t]) where id > 0                 │
│                                                                 │
│  2. For each cell:                                              │
│     ├── Get cell centroid from segmentation (z, y, x)          │
│     ├── CARVE patch from raw image around centroid             │
│     │   └── raw[t-1:t+2, z±4, y±32, x±32]                      │
│     ├── Classify patch → event class                           │
│     └── If positive event (class > 0):                         │
│         └── Record GLOBAL coordinates (t, z, y, x)             │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                         NMS Filtering                           │
│  Remove duplicate detections in space-time                      │
│  └── nms_space_time(detections, space_thresh=10, time_thresh=2)│
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Output CSVs                             │
│  mitosis.csv: t,z,y,x   (global image coordinates)             │
│  apoptosis.csv: t,z,y,x                                        │
└─────────────────────────────────────────────────────────────────┘
```

**Key Concept: Segmentation-Guided Prediction**

The segmentation image serves two purposes:
1. **Localization**: Identifies where cells are in the image
2. **Patch extraction**: Defines regions to carve from raw image

This is different from sliding-window approaches - we only classify regions where cells actually exist, making prediction efficient and biologically meaningful.

```python
# Inside predict_step (simplified):
for cell_id in unique_cells:
    # Get centroid from segmentation
    cell_mask = (seg == cell_id)
    z, y, x = center_of_mass(cell_mask)

    # Carve patch from RAW image (not seg!)
    patch = raw[:, z-4:z+4, y-32:y+32, x-32:x+32]

    # Classify
    event_class = model(patch).argmax()

    # Record global coordinates if positive
    if event_class > 0:
        detections.append({'t': t, 'z': z, 'y': y, 'x': x, 'class': event_class})
```

---

## Model Architecture

### DenseVollNet

3D DenseNet for volumetric event detection with dual output heads.

```
Input: (T, Z, Y, X) patches
         │
    ┌────▼────┐
    │ DenseNet │
    │  Encoder │
    └────┬────┘
         │
    ┌────▼────┐
    │  Dense   │
    │  Blocks  │
    └────┬────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌───▼───┐
│Classes│ │  Box  │
│Softmax│ │Sigmoid│
└───────┘ └───────┘

Output: [categories] + [x,y,z,t,h,w,d,c]
```

**Key Parameters:**
- `input_shape`: (T, Z, Y, X) - temporal window size
- `categories`: Number of event classes
- `box_vector`: 8 (x, y, z, t, h, w, d, c)
- `depth`: DenseBlock depth config
- `growth_rate`: Feature growth rate

---

## Loss Function

### VolumeYoloLoss

Multi-task loss combining classification and localization.

```python
from kapoorlabs_lightning.pytorch_losses import VolumeYoloLoss

loss_func = VolumeYoloLoss(
    categories=2,
    box_vector=8,
    device="cuda",
    class_weights_dict={0: 1.0, 1: 5.0},  # Handle class imbalance
    return_components=True,
)
```

**Components:**
1. **Classification**: NLLLoss on softmax probabilities
2. **Localization (xyzt)**: MSE loss
3. **Dimensions (hwd)**: sqrt-MSE loss (YOLO-style)
4. **Confidence**: MSE loss

**Tensor Format Handling:**
- Ground truth: `[box_vector, categories]` - box first, then one-hot
- Model output: `[categories, box_vector]` - softmax first, then sigmoid

---

## Transform Presets

### Light
```yaml
- Gaussian noise (std=0.01)
- Spatial flips (p=0.3)
```

### Medium
```yaml
- Gaussian noise (std=0.02)
- Poisson noise (p=0.3)
- Gaussian blur (p=0.3)
- Spatial flips (p=0.5)
- 90° rotations (p=0.5)
```

### Heavy
```yaml
- Gaussian noise (std=0.03)
- Poisson noise (p=0.5)
- Gaussian blur (p=0.5)
- Spatial flips (p=0.7)
- 90° rotations (p=0.7)
- Brightness/contrast (p=0.3)
- Elastic deformation (p=0.3)
```

---

## Optimizer Comparison

| Optimizer | Learning Rate | Use Case |
|-----------|---------------|----------|
| **Adam**  | 1e-3 to 1e-4  | Recommended for ONEAT. Better handling of multi-task gradients. |
| SGD       | 0.01 to 0.1   | Requires careful LR scheduling. May need higher LR + momentum. |
| LARS      | 0.1 to 1.0    | Large batch training. |

**Why Adam works better for ONEAT:**
- Multi-task loss has different gradient scales (classification vs regression)
- Adam's per-parameter learning rates adapt automatically
- Sparse event labels benefit from Adam's moment estimates

---

## Configuration

### Model Parameters
```yaml
imagex: 64
imagey: 64
imagez: 8
depth: {depth_0: 12, depth_1: 24, depth_2: 16}
growth_rate: 32
pool_first: True
```

### Temporal Window
```yaml
size_tminus: 1  # frames before event
size_tplus: 1   # frames after event
```

### YOLO Box Vector
```yaml
event_position_label: ["x", "y", "z", "t", "h", "w", "d", "c"]
```

### Normalization
```yaml
normalizeimage: True
pmin: 1.0       # percentile min
pmax: 99.8      # percentile max
```

### NMS
```yaml
nms_space: 10   # spatial distance threshold
nms_time: 2     # temporal distance threshold
```

### Events
```yaml
num_classes: 2
event_name: ['normal', 'mitosis']
```

---

## Directory Structure

```
KapoorLabs-Lightning/
├── scripts/
│   ├── conf/                              # Hydra configs
│   │   ├── parameters/oneat.yaml          # Model & training params
│   │   ├── train_data_paths/gwdg.yaml     # Data paths
│   │   └── scenario_*.yaml                # Scenario configs
│   ├── train_data_generation/
│   │   └── generate-oneat-training-data.py
│   ├── model_training/
│   │   ├── lightning-oneat.py             # SGD training
│   │   ├── lightning-oneat-adam.py        # Adam training
│   │   ├── slurm_train_adam_*.sh          # SLURM scripts
│   │   ├── submit_lr_sweep_*.py           # Hyperparameter sweeps
│   │   └── monitor_training.py            # Checkpoint monitor
│   └── model_prediction/
│       └── lightning-predict-oneat.py
└── src/kapoorlabs_lightning/
    ├── oneat_module.py                    # OneatActionModule
    ├── pytorch_losses.py                  # VolumeYoloLoss
    ├── oneat_prediction_dataset.py        # Prediction dataset
    ├── oneat_transforms.py                # Augmentations
    ├── oneat_presets.py                   # Transform presets
    ├── nms_utils.py                       # Space-time NMS
    ├── pytorch_models.py                  # DenseVollNet
    └── utils.py                           # H5 creation, plotting
```

---

## Training Monitoring

### Real-time Plots
```bash
python monitor_training.py
```
Monitors checkpoint folders every 5 minutes, generates metric plots, and commits to git.

### Metrics Logged
- `train_loss` / `val_loss` - Combined loss
- `train_loss_class` / `val_loss_class` - Classification component
- `train_loss_xyzt_hwd` / `val_loss_xyzt_hwd` - Localization component
- `train_loss_conf` / `val_loss_conf` - Confidence component
- `train_class_accuracy` / `val_class_accuracy` - Classification accuracy
- `learning_rate` - Current LR

---

## Example Configs

### Train
```yaml
# scenario_train_oneat.yaml
defaults:
  - train_data_paths: gwdg.yaml
  - parameters: oneat.yaml

# gwdg.yaml
log_path: 'logs/oneat_training'
oneat_h5_file: 'oneat_data.h5'

# oneat.yaml
num_classes: 2
imagex: 64
imagey: 64
imagez: 8
batch_size: 32
learning_rate: 0.001
epochs: 250
transform_preset: 'medium'
```

### Predict
```yaml
# scenario_predict_oneat.yaml
defaults:
  - experiment_data_paths: dataset.yaml
  - train_data_paths: gwdg.yaml
  - parameters: oneat.yaml

# dataset.yaml
base_data_dir: '/path/to/data/'
raw_timelapses: 'raw_dataset/'
seg_timelapses: 'seg_dataset/'
oneat_predictions: 'results/'
```

---

## Notes

- Only non-normal events (class > 0) are saved in predictions
- Normalization must match between training and prediction
- Chunk size: 50 timepoints (adjustable)
- CSV format: ONEAT compatible (`t,z,y,x`)
- Adam optimizer is recommended over SGD for this multi-task loss

---

## License

BSD-3-Clause

## Citation

If using this software, cite KapoorLabs-Lightning and the original ONEAT paper.

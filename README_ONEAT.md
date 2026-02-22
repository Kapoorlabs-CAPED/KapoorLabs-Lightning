# ONEAT: Action Classification for 3D+Time Microscopy Data

Action classification for TZYX microscopy timelapses using DenseVollNet architecture.

## Installation

```bash
pip install kapoorlabs-lightning
```

## Workflow

### 1. Generate Training Data

```bash
cd scripts/train_data_generation
python generate-oneat-training-data.py
```

**Inputs:**
- Raw TZYX timelapse images (`*.tif`)
- Segmentation masks with cell labels
- CSV files with event annotations (`t,z,y,x`)

**Process:**
- Normalizes raw images in chunks (50 timepoints)
- Extracts temporal patches around annotated events
- Creates train/val split
- Saves to HDF5 file

**Config:** `scripts/conf/scenario_generate_oneat.yaml`

---

### 2. Train Model

```bash
cd scripts/model_training
python lightning-oneat.py
```

**Steps:**
1. Instantiate DenseVollNet model
2. Setup transforms (light/medium/heavy)
3. Load H5 dataset
4. Configure optimizer (Adam/SGD/LARS)
5. Configure scheduler (Cosine/WarmCosine)
6. Train with Lightning

**Transform Presets:**
- `light`: Minimal (noise + flips)
- `medium`: Balanced (+ blur + rotation)
- `heavy`: Aggressive (+ elastic + brightness)

**Config:** `scripts/conf/scenario_train_oneat.yaml`

---

### 3. Predict

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

## Configuration

### Model Architecture
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

## Architecture

**DenseVollNet**: 3D DenseNet for volumetric action classification

- **Input**: (T, Z, Y, X) patches
- **Output**: Class logits + bounding box regression
- **Features**: Dense connections, multi-scale processing

---

## Key Features

✅ **Chunk normalization**: Processes time in blocks, preserves temporal structure
✅ **Lightning integration**: Uses `trainer.predict()` for inference
✅ **NMS space-time**: Filters overlapping detections
✅ **Argmax prediction**: No confidence threshold
✅ **Memory efficient**: Streams H5 data, batch processing
✅ **Hydra config**: All parameters in YAML

---

## Directory Structure

```
KapoorLabs-Lightning/
├── scripts/
│   ├── conf/                          # Hydra configs
│   │   ├── parameters/oneat.yaml      # Model & training params
│   │   ├── train_data_paths/gwdg.yaml # Data paths
│   │   └── scenario_*.yaml            # Scenario configs
│   ├── train_data_generation/
│   │   └── generate-oneat-training-data.py
│   ├── model_training/
│   │   └── lightning-oneat.py
│   └── model_prediction/
│       └── lightning-predict-oneat.py
└── src/kapoorlabs_lightning/
    ├── oneat_module.py               # OneatActionModule with predict_step
    ├── oneat_prediction_dataset.py   # Prediction dataset
    ├── nms_utils.py                  # NMS in space-time
    └── pytorch_models.py             # DenseVollNet
```

---

## Example Config

**Train:**
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
learning_rate: 0.25
epochs: 250
transform_preset: 'medium'
```

**Predict:**
```yaml
# scenario_predict_oneat.yaml
defaults:
  - experiment_data_paths: jean_zay_first_dataset.yaml
  - train_data_paths: gwdg.yaml
  - parameters: oneat.yaml

# jean_zay_first_dataset.yaml
base_data_dir: '/path/to/data/'
raw_timelapses: 'raw_dataset/'
seg_timelapses: 'seg_dataset/'
oneat_predictions: 'results/'
```

---

## Notes

- Only non-normal events (class > 0) are saved
- Normalization must match between training and prediction
- Chunk size: 50 timepoints (adjustable)
- CSV format: ONEAT compatible (`t,z,y,x`)

---

## License

BSD-3-Clause

## Citation

If using this software, cite KapoorLabs-Lightning and the original ONEAT paper.

# CARE: Content-Aware image REstoration for 3D Microscopy

Supervised 3D denoising using paired low/high SNR images, implemented with the UNet architecture from [CAREamics](https://github.com/CAREamics/careamics) and trained via PyTorch Lightning.

## Installation

```bash
pip install kapoorlabs-lightning careamics
```

## Workflow Overview

```
1. Generate H5 Dataset    →    2. Train Model    →    3. Predict (denoise volumes)
   (paired patches)             (CareInception)        (tiled + overlap blending)
```

---

## 1. Generate Training Data

```bash
cd scripts/train_data_generation
python generate-care-training-data.py
```

**Inputs:**
- `low/` directory — low SNR `.tif` files (ZYX or CZYX)
- `high/` directory — high SNR `.tif` files (same filenames as `low/`)

**Process:**
1. Pairs matching filenames across `low/` and `high/`
2. Reserves the last file for validation; all others for training
3. Extracts 3D patches with configurable shape and stride (default 2/3 overlap for train, non-overlapping for val)
4. Streams patches directly to H5 in batches — no large arrays held in memory

**Output H5 Structure:**
```
care_training_data.h5
├── train/
│   ├── low     (N, patch_z, patch_y, patch_x) float32
│   └── high    (N, patch_z, patch_y, patch_x) float32
└── val/
    ├── low
    └── high
```

**Configuration** (`scripts/conf/parameters/care.yaml`):
```yaml
patch_z: 16
patch_y: 64
patch_x: 64
file_type: '*.tif'
```

---

## 2. Train the Model

```bash
cd scripts/model_training
python lightning-care.py
```

Uses `CareInception` — an orchestrator class that stacks all setup steps cleanly:

```python
from kapoorlabs_lightning import CareInception

trainer = CareInception(
    h5_file="care_training_data.h5",
    epochs=100,
    batch_size=16,
    learning_rate=4e-4,
    n_tiles=[1, 4, 4],
    tile_overlap=0.125,
    slurm_auto_requeue=True,   # auto-requeue on HPC clusters
)

# Pick a transform preset: light / medium / heavy
trainer.setup_care_transforms_medium(
    pmin=0.1, pmax=99.9,
    spatial_flip_p=0.5,
    rotation_p=0.5,
    gaussian_noise_std=0.01,
)

trainer.setup_care_h5_datasets()

trainer.setup_care_unet_model(
    unet_depth=3,
    num_channels_init=64,
    use_batch_norm=True,
)

trainer.setup_adam()                    # or setup_adamw() / setup_sgd()
trainer.setup_learning_rate_scheduler()
trainer.setup_care_lightning_model()
trainer.train(logger=logger, callbacks=callbacks)
```

**UNet Architecture:**
- Sourced from `careamics.models.unet.UNet`
- 3D convolutions (`conv_dims=3`)
- Single input channel, single output channel
- Depth and initial filter count configurable

**Loss:** MSE (standard for CARE supervised denoising)

**Logged Metrics:** `train_loss`, `val_loss`, `val_psnr`

**Output:**
- Checkpoints saved to `log_path/`
- Hyperparameters saved as `{experiment_name}.json`

---

## 3. Predict (Denoise Volumes)

```bash
cd scripts/model_prediction
python predict-care.py
```

Prediction uses tiled inference with linear-blend overlap stitching to handle volumes larger than the training patch size.

```python
from kapoorlabs_lightning import CareModule, CarePredictionDataset, stitch_tiles

# Load model from checkpoint
model = CareModule.load_checkpoint(
    network=unet,
    loss_func=loss,
    optim_func=optimizer,
    checkpoint_dir="/path/to/checkpoints/",
)

# Tile a 3D volume and run batched inference
dataset = CarePredictionDataset(
    volume=raw_volume,           # (Z, Y, X) numpy array
    n_tiles=[1, 4, 4],
    overlap=0.125,
    transforms=eval_transforms,
)

# Predict and stitch
tiles_out = []
for batch in DataLoader(dataset, batch_size=4):
    tiles_out.append(model(batch).cpu().numpy())

denoised = stitch_tiles(tiles_out, dataset.tiles_info, raw_volume.shape)
```

---

## Transform Presets

All presets apply geometric transforms identically to both low and high patches (synchronized). Gaussian noise is added to the low (input) channel only.

| Preset | Flip | Rotation | Gaussian Noise |
|--------|------|----------|----------------|
| Light  | ✓    | —        | —              |
| Medium | ✓    | ✓        | std=0.01       |
| Heavy  | ✓    | ✓        | std=0.03       |
| Eval   | —    | —        | —              |

All presets include `PairedPercentileNormalize` (default pmin=0.1, pmax=99.9) and `PairedToFloat32`.

---

## SLURM

```bash
# Generate training data
sbatch scripts/train_data_generation/slurm_generate_care.sh

# Train
sbatch scripts/model_training/slurm_train_care.sh

# Predict
sbatch scripts/model_prediction/slurm_predict_care.sh
```

SLURM training script uses `slurm_auto_requeue=True` in `CareInception`, which signals the Lightning trainer to handle SIGUSR1 for automatic job requeue on preemption.

---

## Visualize Training Data

Open `scripts/train_data_generation/care_h5_visualizer.ipynb` to interactively inspect patches:

- Side-by-side Low SNR / High SNR view
- Difference map (High − Low) with colorbar
- Per-patch statistics: min, max, mean, std, MSE
- Sliders for split, sample index, and Z slice

---

## Configuration Files

| File | Purpose |
|------|---------|
| `scripts/conf/parameters/care.yaml` | UNet architecture, patch shape, training hyperparameters |
| `scripts/conf/train_data_paths/care_default.yaml` | Paths to low/, high/, H5 output, log directory |
| `scripts/conf/scenario_train_care.yaml` | Top-level Hydra config for training |
| `scripts/conf/scenario_generate_care.yaml` | Top-level Hydra config for data generation |
| `scripts/conf/scenario_predict_care.yaml` | Top-level Hydra config for prediction |

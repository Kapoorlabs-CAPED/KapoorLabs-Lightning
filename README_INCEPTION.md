# Inception Cell-Fate Model: Training to Prediction

End-to-end pipeline for the 1D-DenseNet+attention ("InceptionNet") cell-fate
classifier — from a TrackMate XML coming out of NapaTrackMater through training
to per-track prediction and streamlit visualisation.

This is the *current* shape of the pipeline. Where the older
`README_CELLFATE.md` drifted from the code, this file is authoritative.

---

## Pipeline

```
[NapaTrackMater]   →   [TrackVectors]   →   [generate-cellfate-training-data]
   master XML            features_*.csv          cellfate_nuclei.h5
                                                 (N, 25, 18) z-scored
                                                       │
                                                       ▼
                              [lightning-cellfate.py]  trains InceptionNet
                                                       │
                                                       ▼
                                          cellfate_model_<size>/*.ckpt
                                          + training_config.json
                                                       │
                                                       ▼
   master XML  →  [predict-cellfate.py]  →  per-class CSVs + CMs + transitions
   (inference)         (z-scores per dataset)
                                                       │
                                                       ▼
                              [apps/streamlit/inception/remote_app.py]
                              SLURM job submit + result viewer
```

Channels (1D over T=25):

| Index | Name | Group | Source |
| --- | --- | --- | --- |
| 0 | Radius | SHAPE | TrackMate spot radius |
| 1-3 | Eccentricity_Comp_{First,Second,Third} | SHAPE | 3D point-cloud PCA |
| 4 | Local_Cell_Density | SHAPE | neighbour count |
| 5 | Surface_Area | SHAPE | marching-cubes |
| 6 | **Speed** | DYNAMIC | finite-diff of position |
| 7-9 | Motion_Angle_{Z,Y,X} | DYNAMIC | motion vector |
| 10 | Acceleration | DYNAMIC | second-diff of position |
| 11 | Distance_Cell_mask | DYNAMIC | distance to mask boundary |
| 12-14 | Radial_Angle_{Z,Y,X} | DYNAMIC | radial vector |
| 15-17 | Cell_Axis_{Z,Y,X} | DYNAMIC | shape axis from PCA |

`MSD` is *not* a model input — it lives in `TRACK_TYPE_FEATURES` and only shows
up in the streamlit feature-distribution panel.

Canonical lists: `src/kapoorlabs_lightning/tracking/track_features.py`
(`SHAPE_FEATURES`, `DYNAMIC_FEATURES`, `SHAPE_DYNAMIC_FEATURES`). Identical to
the lists NapaTrackMater (`Trackvector.py`) was originally built around.

---

## 1. Source XML (NapaTrackMater)

`predict-cellfate.py` and the H5 builder both read a TrackMate-format XML
through `TrackVectors` (in `src/kapoorlabs_lightning/tracking/`).

Two flavours:

- **Master XML** — Spot nodes carry pre-computed feature attributes
  (`cloud_eccentricity_comp_first`, `acceleration`, `motion_angle_*`,
  `cloud_surfacearea`, `cell_axis_*_key`, …). Detected by the presence of
  `unique_id` on any spot. `TrackVectors._master_dataframe()` reads attributes
  straight from the XML — fast.
- **Plain TrackMate XML + segmentation** — Features are recomputed from
  positions + a seg/mask TIFF via `TrackVectors._dataframe()`.

### What `_master_dataframe` expects on each Spot

The full attribute list lives in `xml_parser.py::TrackMateXML._MASTER_ATTRS`.
Missing keys silently fall back to defaults — for most they default to NaN,
**for `speed` and `msd` to position-derived values** (patched fallback added
2026-05-21; see *Speed channel pitfall* below).

### Speed channel pitfall

Older NapaTrackMater exports drop `speed` and `msd` from Spot nodes but keep
`unique_id` / `acceleration` / `motion_angle_*`. Without the fallback the
features CSV ends up with Speed = 0 everywhere, and since
`predict_all_tracks` z-scores the *current* dataset's Speed column,
`std == 0` collapses Speed to a flat zero into the model — a **dead channel**.

The fix is the per-tracklet position-based recomputation in
`_master_dataframe` (computes `compute_speed(pos, prev_pos, calibration) / dt`
and `compute_msd(positions_phys)` whenever the keys aren't on the spot).

Diagnose with a single grep on the XML:

```bash
grep -c ' speed="' your.xml   # should be >0; if 0, you'll get the dead channel
```

If your XML is old, either re-export with current NapaTrackMater or delete the
cached `features_*.csv` next to the XML so the patched `TrackVectors`
regenerates with computed Speed/MSD.

---

## 2. Features CSV (cached side-car)

`predict-cellfate.py::load_dataframe` caches the feature DataFrame next to
the XML as `features_<xml_stem>.csv` (one row per `(track, frame)`, ~36
columns including identity + the 18 SHAPE_DYNAMIC channels + MSD).

```
features_<stem>.csv          (~100 MB for ~3K tracks × 192 frames)
columns: Track_ID, TrackMate_Track_ID, Generation_ID, Tracklet_Number,
         Unique_ID, t, z, y, x, Dividing, Number_Dividing,
         Radius, Eccentricity_Comp_First, ..., Cell_Axis_X,
         MSD, Track_Displacement, Total_Track_Distance, ...
```

Reuse is controlled by `experiment_data_paths.reuse_cached_features` (default
True). Delete the file to force a rebuild — this is the right move after any
XML or `TrackVectors` change.

---

## 3. Build the training H5

```bash
cd scripts/train_data_generation
python generate-cellfate-training-data.py
```

What it does:

1. Loads the features DataFrame (CSV or XML via TrackVectors).
2. Calls `create_training_tracklets(df, tracklet_length=25, stride=...)` —
   sliding-window cuts each track into `(25, 18)` matrices.
3. **Per-column z-score normalisation** across all training samples (using
   `master_tracking.normalize_features` or equivalent). This is what produces
   the mean ≈ 0, std ≈ 1 distributions you see in `train_arrays`.
4. Train/val split via `sklearn.model_selection.train_test_split`.
5. Writes:

```
cellfate_nuclei.h5
├── train_arrays    (N_train, 25, 18) float64
├── train_labels    (N_train,) int64
├── val_arrays      (N_val,   25, 18) float64
└── val_labels      (N_val,)   int64
```

Inspect with `scripts/train_data_generation/h5_inspect.py` — sanity-check
per-channel means/stds before training:

```bash
python h5_inspect.py /path/to/cellfate_nuclei.h5
```

A healthy split has all 18 channels showing real distributions (no all-zero
columns). The training H5 currently used (`inception_cell_type_nuclei_.h5`)
gives `Speed` min=-1.61, max=43.7, mean=-0.088, std=0.896 with **zero exact
zeros across 8.2M val entries** — confirmed live training input.

---

## 4. Train the model

```bash
cd scripts/model_training
python lightning-cellfate.py
```

Hydra-driven. Defaults come from `scripts/conf/scenario_train_cellfate.yaml`
which composes:

- `parameters: cellfate.yaml` — model + training knobs
- `train_data_paths: cellfate_default.yaml` (override per cluster)

### Architecture

`InceptionNet` (`src/kapoorlabs_lightning/pytorch_models.py:147`) — 1D DenseNet
encoder + attention-pool head:

```
input  (B, 18, 25)
  → Conv1d stem (kernel=7) → BN → ReLU
  → DenseBlock stages (block_config layers each)
    + TransitionBlocks between stages
  → AttentionPool1d (attn_heads=8, learnable CLS + pos embeds)
  → Linear → logits (B, num_classes)
```

`block_config` is `[6]` by default (single dense stage of 6 layers) — small
and fast. `[6, 12, 24, 16]` is the canonical "DenseNet-121-ish" preset.

### Common overrides

```bash
# Transform presets (order-preserving — no time shuffling)
python lightning-cellfate.py parameters.transform_preset=light
python lightning-cellfate.py parameters.transform_preset=medium  # default
python lightning-cellfate.py parameters.transform_preset=heavy

# Backbone choice
python lightning-cellfate.py parameters.model_choice=inception   # default
python lightning-cellfate.py parameters.model_choice=densenet
python lightning-cellfate.py parameters.model_choice=mitosisnet
```

### Scheduler

Hydra config group at `scripts/conf/parameters/scheduler/`. Default
`WarmCosineAnnealingLR`; override e.g. `parameters/scheduler=cosine`.

### Checkpoint side-cars

`log_path/` after training holds:

```
cellfate_model_<size>/
├── epoch=120-step=40172.ckpt        # Lightning checkpoint
├── training_config.json             # Hydra config dump (read at inference)
├── cellfate_nuclei.json             # NPZ logger metadata
└── cellfate_nuclei.npz              # per-epoch metrics
```

`training_config.json` is critical — `predict-cellfate.py` consults it
via `_arch_loader.load_arch_from_training_config` and uses its arch keys
*ahead* of the prediction yaml, so the same prediction script handles
checkpoints trained with different architectures.

---

## 5. Predict

```bash
cd scripts/model_prediction
python predict-cellfate.py \
    experiment_data_paths.xml_file=/path/to/master.xml \
    experiment_data_paths.checkpoint_path=/path/to/cellfate_model_medium \
    experiment_data_paths.output_dir=/path/to/results \
    parameters.input_mode=xml \
    parameters.tracklet_length=25 \
    parameters.time_window="[0,-1]"
```

What runs (in order):

1. **Load arch** from the checkpoint's `training_config.json`; build
   `InceptionNet` / `DenseNet` / `MitosisNet` matching how it was trained.
2. **Load weights** from the latest `.ckpt` in the checkpoint dir.
3. **Build the feature DataFrame** from XML (via `TrackVectors`) or reuse
   `features_*.csv` if present and `reuse_cached_features=true`.
4. **Time-window truncation** if `parameters.time_window` is set
   (`[start, end]`, inclusive, frame units; `end=-1` means last frame).
   Span must be > 50.
5. **Z-score normalise** every feature column on *this* dataset's stats
   (`track_prediction.py:175-187`). No frozen training stats — the model
   was trained on z-scored data and is fed z-scored data at inference.
6. **`predict_all_tracks`** — two-level voting:
   - Each track is cut into 25-frame tracklets.
   - Each tracklet → softmax → argmax → class vote.
   - Per-parent-track majority vote → final predicted class.
7. **Write outputs** to `output_dir`:

```
<prefix>basal_predictions.csv          one row per predicted Basal track (first t,z,y,x)
<prefix>goblet_predictions.csv
<prefix>radial_predictions.csv
<prefix>all_predictions.csv            TrackMate_Track_ID, Predicted_Class
<prefix>confusion_matrix.csv           global CM (true rows × pred cols)
<prefix>confusion_matrix.png
<prefix>t<a>-<b>_confusion_matrix.csv  per-block CMs (sliding window over time)
<prefix>t<a>-<b>_confusion_matrix.png
<prefix>gt_track_assignments.csv       track_id → GT_Class (from nearest-neighbour match)
<prefix>transition_confidence.{csv,png}  sliding-window prediction stability
<prefix>transition_refined.csv         coarse-to-fine refined transition windows
```

`<prefix>` is `output_prefix + window_tag`, e.g. `t0-191_`.

### Knobs you'll actually touch

| key | meaning | default |
| --- | --- | --- |
| `parameters.tracklet_length` | window length the model classifies | 25 |
| `parameters.time_window` | `[start, end]`, span > 50 frames | `[100, -1]` |
| `parameters.transition_time_determination` | sweep stability windows | true |
| `parameters.per_block_cm` | also emit per-block CMs | true |
| `parameters.cm_window_span` / `cm_window_stride` | sliding-CM geometry | 50 / 25 |
| `parameters.input_mode` | `xml` or `csv` | xml |
| `parameters.accelerator` | `cuda` / `cpu` | cuda |

GT csvs in the yaml drive the confusion matrix: set
`experiment_data_paths.{basal,goblet,radially_intercalating}_gt_annotations`
to the per-class GT CSV paths (columns `T,Z,Y,X`). Without them you get
predictions but no CMs.

---

## 6. Streamlit viewer

`apps/streamlit/inception/remote_app.py` — the user-facing app:

- ORCID login + per-user quota
- Pick a curated demo dataset (under `JEANZAY_MOUNT/uploads/<name>/`) and a
  checkpoint dir (`MODELS_DIR/cellfate_model_*`)
- Submits a SLURM job to Jean Zay (via SSH) that runs the predict script
  exactly as above
- Polls `RESULTS_DIR/<job_id>/status.txt`; once `DONE`, renders three tabs:

| Tab | What it shows |
| --- | --- |
| Confusion matrix | Per-block CM with a numeric-sorted slider over time blocks (`t0-49`, `t25-74`, …). Global accuracy line. |
| Predicted track IDs | Per-track feature distributions: GT (left, violins by class) vs Predicted (right). Plus download buttons for every prediction CSV. |
| Timelapse viewer | Max-Z projection of the demo TIFF at a chosen frame, with GT cells overlaid (open circles, per-class colours) and predicted cells (×, same palette). Pixel-space — no calibration division needed since both GT and feature CSVs are pixel-native. |

Default checkpoint selection picks the `*_medium` model when present. Coords
in CSVs are already in pixel space; calibration division was a misdiagnosis
from an earlier iteration and would push points outside the image.

---

## File map

```
KapoorLabs-Lightning/
├── README_INCEPTION.md                      # this file
├── README_CELLFATE.md                       # older, partial — superseded here
├── src/kapoorlabs_lightning/
│   ├── pytorch_models.py                    # InceptionNet, DenseNet, MitosisNet
│   ├── cellfate_module.py                   # Lightning module
│   ├── lightning_trainer.py                 # MitosisInception trainer
│   ├── time_series_presets.py               # transform presets (light/medium/heavy)
│   └── tracking/
│       ├── track_features.py                # SHAPE_/DYNAMIC_/SHAPE_DYNAMIC_FEATURES
│       ├── track_vectors.py                 # TrackVectors (_dataframe, _master_dataframe)
│       ├── track_prediction.py              # predict_all_tracks + z-score normalisation
│       ├── xml_parser.py                    # TrackMateXML + _MASTER_ATTRS
│       └── master_graph.py                  # enrich_graph_with_dynamics
├── scripts/
│   ├── conf/
│   │   ├── parameters/cellfate.yaml         # training knobs
│   │   ├── parameters/cellfate_predict.yaml # prediction knobs (incl. per_block_cm)
│   │   ├── parameters/cellfate_datagen.yaml # H5 build knobs
│   │   ├── parameters/scheduler/*.yaml      # WarmCosineAnnealingLR etc.
│   │   ├── experiment_data_paths/cellfate_predict_jeanzay.yaml
│   │   └── scenario_{train,predict}_cellfate*.yaml
│   ├── train_data_generation/
│   │   ├── generate-cellfate-training-data.py
│   │   └── h5_inspect.py
│   ├── model_training/
│   │   └── lightning-cellfate.py
│   └── model_prediction/
│       ├── predict-cellfate.py
│       ├── _arch_loader.py                  # reads training_config.json
│       └── scenario_predict_cellfate_inception.py
└── apps/streamlit/inception/
    ├── remote_app.py
    ├── submit_inception_job.sh
    └── run_inception_job.sh                 # runs inside the SLURM allocation
```

---

## Troubleshooting

**`No module named '_arch_loader'`** — file was accidentally deleted; restore
from git (`scripts/model_prediction/_arch_loader.py`). Both
`predict-cellfate.py` and `predict-oneat.py` import it.

**Speed and/or MSD violins are flat at 0** — the features cache was built
from an XML missing `speed` / `msd` attrs and predates the
`_master_dataframe` fallback patch. Delete the cached `features_*.csv` next
to the XML and re-run the prediction job.

**`No confusion matrix in results dir`** — `predict-cellfate.py` only writes
the CM when `experiment_data_paths.{basal,goblet,radially_intercalating}_gt_annotations`
all point at on-disk CSVs. Check the SLURM log; missing-GT prints
`No GT annotation files found; skipping confusion matrix.`

**Only one CM despite `per_block_cm: true`** — your `time_window` span is
≤ `cm_window_span`. Lower `cm_window_span` (e.g. 25) and/or stride to get
more blocks.

**Predictions look random** — check the features CSV: any all-zero column
means a dead input channel after z-scoring. Most often Speed or one of the
shape components missing from the XML.

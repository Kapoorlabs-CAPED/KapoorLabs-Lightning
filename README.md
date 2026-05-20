# KapoorLabs-Lightning



[![License BSD-3](https://img.shields.io/pypi/l/KapoorLabs-Lightning.svg?color=green)](https://github.com/Kapoorlabs-CAPED/KapoorLabs-Lightning/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/KapoorLabs-Lightning.svg?color=green)](https://pypi.org/project/KapoorLabs-Lightning)
[![Python Version](https://img.shields.io/pypi/pyversions/KapoorLabs-Lightning.svg?color=green)](https://python.org)
[![codecov](https://codecov.io/gh/Kapoorlabs-CAPED/KapoorLabs-Lightning/branch/main/graph/badge.svg)](https://codecov.io/gh/Kapoorlabs-CAPED/KapoorLabs-Lightning)




## Lightning Modules for KapoorLabs Projects

PyTorch Lightning framework for training deep learning models on microscopy data, with specialized support for:
- **ONEAT**: Spatio-temporal event detection in 3D+T microscopy data
- **Cell Fate Classification**: Time series classification of cell fates (basal, goblet, radial) from tracking data
- **CARE**: Content-Aware image REstoration — supervised 3D denoising with paired low/high SNR training data
- **Tracking bridge**: Convert [Trackastra](https://github.com/weigertlab/trackastra) `networkx.DiGraph` output into the same DataFrame the TrackMate path produces, with Oneat-driven division correction and a "master corrected graph" that mirrors NapaTrackMater's `master_<original>.xml`

----------------------------------

## Key Features

- **Modular Architecture**: Base, ONEAT, Cell Fate, and CARE Lightning modules
- **YOLO-style Detection**: VolumeYoloLoss for multi-task learning (classification + localization)
- **H5 Dataset Support**: Memory-efficient streaming from HDF5 files — patches written incrementally, never held in memory
- **Segmentation-Guided Prediction**: Uses instance segmentation to locate cells, carves patches from raw image, classifies each cell, and records global coordinates for positive events
- **CARE Denoising**: Supervised 3D denoising via UNet (careamics), tiled prediction with linear-blend overlap stitching
- **Transform Presets**: Light, Medium, Heavy augmentation pipelines for microscopy data (including paired transforms for denoising)
- **Multiple Optimizers**: Adam, SGD, LARS, AdamW with learning rate schedulers
- **SLURM Integration**: Auto-requeue support for HPC clusters
- **Hydra Configuration**: YAML-based experiment configuration
- **Trackastra → KapoorLabs DataFrame**: graph-bridge + Oneat correction so cell-fate / inception / curvature ML stacks consume Trackastra and TrackMate output through one schema

## Package Structure

```
kapoorlabs_lightning/
├── Lightning Modules
│   ├── base_module.py          # BaseModule - common functionality
│   ├── oneat_module.py         # OneatActionModule - event detection
│   ├── cellfate_module.py      # CellFateModule - time series classification
│   └── care_module.py          # CareModule - 3D denoising (MSE + PSNR, tiled predict)
├── Models
│   ├── pytorch_models.py       # DenseVollNet, DenseNet, InceptionNet
│   └── pytorch_losses.py       # VolumeYoloLoss, OneatClassificationLoss
├── Data
│   ├── pytorch_datasets.py        # H5VisionDataset, H5MitosisDataset
│   ├── oneat_prediction_dataset.py # OneatPredictionDataset (seg-guided inference)
│   └── care_dataset.py            # H5CareDataset, CarePredictionDataset
├── Transforms
│   ├── oneat_transforms.py     # Microscopy-specific augmentations
│   ├── oneat_presets.py        # Light/Medium/Heavy presets
│   ├── time_series_presets.py  # Cell fate transforms + presets (order-preserving)
│   ├── care_transforms.py      # Paired transforms for denoising (low+high in sync)
│   └── care_presets.py         # CARE Light/Medium/Heavy/Eval presets
├── Training
│   ├── lightning_trainer.py    # MitosisInception trainer class
│   ├── care_trainer.py         # CareInception trainer class
│   ├── optimizers.py           # Adam, SGD, LARS, AdamW
│   └── schedulers.py           # Cosine, WarmCosine, Step
├── Tracking
│   ├── xml_parser.py           # TrackMateXML reader (already-corrected XML)
│   ├── xml_writer.py           # write_trackmate_xml → master_<original>.xml
│   ├── track_vectors.py        # TrackVectors._master_dataframe (TrackMate fast path)
│   ├── track_features.py       # compute_speed/msd/angles + feature constants
│   ├── trackastra_bridge.py    # walk_tracklets / graph_to_dataframe / dataframe_to_graph
│   ├── oneat_graph_correction.py  # apply Oneat CSV → repair missed divisions on a DiGraph
│   └── master_graph.py         # enrich + write_master_graph (graph analogue of master XML)
├── Utilities
│   ├── utils.py                # H5 creation, normalization, plotting
│   ├── nms_utils.py            # Space-time NMS
│   └── pytorch_callbacks.py    # Checkpointing, progress bars
└── Logging
    └── pytorch_loggers.py      # CustomNPZLogger for metrics
```

## Installation

You can install `KapoorLabs-Lightning` via [pip]:

    pip install KapoorLabs-Lightning



To install latest development version :

    pip install git+https://github.com/Kapoorlabs-CAPED/KapoorLabs-Lightning.git

## Documentation

- [ONEAT Training Guide](README_ONEAT.md) - Complete workflow for event detection
- [Cell Fate Classification Guide](README_CELLFATE.md) - Time series cell fate classification
- [CARE Denoising Guide](README_CARE.md) - 3D supervised denoising workflow
- [Lightning Modules](src/kapoorlabs_lightning/README_litmodules.md) - Module architecture details
- [Tracking bridge & Oneat graph correction](#trackastra-bridge--oneat-graph-correction) — Trackastra → DataFrame, division repair without TrackMate

## Quick Start

### ONEAT Event Detection

```python
from kapoorlabs_lightning import MitosisInception

# Initialize trainer
trainer = MitosisInception(
    h5_file="training_data.h5",
    num_classes=2,
    epochs=100,
    batch_size=32,
    learning_rate=1e-3,
)

# Setup model and training
trainer.setup_densenet_vision_model(
    input_shape=(3, 8, 64, 64),  # (T, Z, Y, X)
    categories=2,
    box_vector=8,
)
trainer.setup_oneat_transforms_medium()
trainer.setup_vision_h5_datasets()
trainer.setup_adam()
trainer.setup_oneat_lightning_model()
trainer.train()
```

### Cell Fate Classification

```python
from kapoorlabs_lightning import MitosisInception

# Initialize trainer
trainer = MitosisInception(
    h5_file="cellfate_data.h5",  # H5 with train_arrays/train_labels/val_arrays/val_labels
    num_classes=3,               # e.g. basal, goblet, radial
    epochs=250,
    batch_size=64,
    learning_rate=1e-3,
    seq_len=25,                  # 25 timepoints per track
)

# Setup (no temporal order changes in transforms)
trainer.setup_cellfate_transforms_medium()
trainer.setup_gbr_h5_datasets()
trainer.setup_inception_qkv_model()
trainer.setup_adam()
trainer.setup_cellfate_lightning_model()
trainer.train()
```

### Trackastra bridge & Oneat graph correction

The tracking module lets cell-fate / inception / curvature downstream code consume either a TrackMate-Oneat-corrected XML **or** a Trackastra `networkx.DiGraph` through one schema. The Trackastra path mirrors what NapaTrackMater does on the XML side — Fiji-edited XML ⇒ `master_<original>.xml` ⇒ DataFrame — but the editing happens in Python on the graph instead of in Fiji.

**Pipeline (Trackastra side):**

```
trackastra.Trackastra().track(imgs, masks)               → nx.DiGraph
    │
    ▼ oneat_correct_graph(G, oneat_csv)                  ← add missed divisions
    ▼ enrich_graph_with_shape_features(G, seg, raw)      ← shape + intensity cached on nodes
    ▼ enrich_graph_with_dynamics(G, calibration)         ← Speed/Acc/Angles/MSD/track-aggs cached
    ▼ write_master_graph(G, "master.json")               ← persisted (analogue of master_*.xml)
    ▼ read_master_graph(path)                             ← reload, no seg/raw needed
    ▼ graph_to_dataframe(G)                               ← fast path: reads cached attrs
    │
    ▼ cellfate / oneat training / curvature scripts (unchanged)
```

**Example:**

```python
from kapoorlabs_lightning.tracking import (
    oneat_correct_graph,
    enrich_graph_with_shape_features, enrich_graph_with_dynamics,
    write_master_graph, read_master_graph, graph_to_dataframe,
)

# Stage 1 — repair divisions Trackastra missed using an Oneat events CSV
G, audit = oneat_correct_graph(
    trackastra_graph,
    "oneat_Division_movie.csv",
    calibration=(2.0, 0.69, 0.69),
    max_match_distance=10.0, max_daughter_distance=20.0,
)

# Stage 2 — master enrichment: per-spot shape, intensity, dynamics cached on nodes
G = enrich_graph_with_shape_features(G, seg_image=seg, raw_image=raw,
                                     calibration=(2.0, 0.69, 0.69))
G = enrich_graph_with_dynamics(G, calibration=(2.0, 0.69, 0.69))

# Stage 3 — persist; the JSON plays the role of master_<original>.xml
write_master_graph(G, "master_movie.json")
G_reloaded = read_master_graph("master_movie.json")

# Stage 4 — same DataFrame schema as TrackVectors.to_dataframe()
df = graph_to_dataframe(G_reloaded)
# Columns: Track_ID, TrackMate_Track_ID, Generation_ID, Tracklet_Number,
#          t, z, y, x, Dividing, Number_Dividing, Radius, Eccentricity_Comp_*,
#          Speed, Acceleration, Motion_Angle_*, MSD, Track_Displacement, ...
```

`TrackMate_Track_ID` is preserved as a first-class column on the Trackastra path (it maps to the connected-component id of the graph), so the same cell-fate / oneat training scripts work with either tracker. To round-trip Oneat-corrected DataFrames back to a Trackastra-shaped graph (for the Trackastra napari viewer, ILP refits, or `apply_solution_graph_to_masks`), use `dataframe_to_graph(df)`.

### CARE 3D Denoising

```python
from kapoorlabs_lightning import CareInception

trainer = CareInception(
    h5_file="care_training_data.h5",
    epochs=100,
    batch_size=16,
    learning_rate=4e-4,
    n_tiles=[1, 4, 4],
    tile_overlap=0.125,
)

trainer.setup_care_transforms_medium()
trainer.setup_care_h5_datasets()
trainer.setup_care_unet_model(unet_depth=3, num_channels_init=64)
trainer.setup_adam()
trainer.setup_learning_rate_scheduler()
trainer.setup_care_lightning_model()
trainer.train(logger=logger, callbacks=callbacks)
```

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"KapoorLabs-Lightning" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.


[pip]: https://pypi.org/project/pip/
[caped]: https://github.com/Kapoorlabs-CAPED
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@caped]: https://github.com/Kapoorlabs-CAPED
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-template]: https://github.com/Kapoorlabs-CAPED/cookiecutter-template

[file an issue]: https://github.com/Kapoorlabs-CAPED/KapoorLabs-Lightning/issues

[caped]: https://github.com/Kapoorlabs-CAPED/
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

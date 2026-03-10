# KapoorLabs-Lightning

## Developed by KapoorLabs


<img src="images/mtrack.png" alt="Logo1" width="150"/>
<img src="images/kapoorlablogo.png" alt="Logo2" width="150"/>

This product is a testament to our expertise at KapoorLabs, where we specialize in creating cutting-edge solutions. We offer bespoke pipeline development services, transforming your developmental biology questions into publishable figures with our advanced computer vision and AI tools. Leverage our expertise and resources to achieve end-to-end solutions that make your research stand out.

**Note:** The tools and pipelines showcased here represent only a fraction of what we can achieve. For tailored and comprehensive solutions beyond what was done in the referenced publication, engage with us directly. Our team is ready to provide the expertise and custom development you need to take your research to the next level. Visit us at [KapoorLabs](https://www.kapoorlabs.org/).



[![License BSD-3](https://img.shields.io/pypi/l/KapoorLabs-Lightning.svg?color=green)](https://github.com/Kapoorlabs-CAPED/KapoorLabs-Lightning/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/KapoorLabs-Lightning.svg?color=green)](https://pypi.org/project/KapoorLabs-Lightning)
[![Python Version](https://img.shields.io/pypi/pyversions/KapoorLabs-Lightning.svg?color=green)](https://python.org)
[![codecov](https://codecov.io/gh/Kapoorlabs-CAPED/KapoorLabs-Lightning/branch/main/graph/badge.svg)](https://codecov.io/gh/Kapoorlabs-CAPED/KapoorLabs-Lightning)




## Lightning Modules for KapoorLabs Projects

PyTorch Lightning framework for training deep learning models on microscopy data, with specialized support for:
- **ONEAT**: Spatio-temporal event detection in 3D+T microscopy data
- **Cell Fate Classification**: Time series classification of cell fates (basal, goblet, radial) from tracking data

----------------------------------

## Key Features

- **Modular Architecture**: Base, ONEAT, and Cell Fate Lightning modules
- **YOLO-style Detection**: VolumeYoloLoss for multi-task learning (classification + localization)
- **H5 Dataset Support**: Memory-efficient streaming from HDF5 files with YOLO labels
- **Segmentation-Guided Prediction**: Uses instance segmentation to locate cells, carves patches from raw image, classifies each cell, and records global coordinates for positive events
- **Transform Presets**: Light, Medium, Heavy augmentation pipelines for microscopy data
- **Multiple Optimizers**: Adam, SGD, LARS, AdamW with learning rate schedulers
- **SLURM Integration**: Auto-requeue support for HPC clusters
- **Hydra Configuration**: YAML-based experiment configuration

## Package Structure

```
kapoorlabs_lightning/
├── Lightning Modules
│   ├── base_module.py          # BaseModule - common functionality
│   ├── oneat_module.py         # OneatActionModule - event detection
│   └── cellfate_module.py      # CellFateModule - time series classification
├── Models
│   ├── pytorch_models.py       # DenseVollNet, DenseNet, InceptionNet
│   └── pytorch_losses.py       # VolumeYoloLoss, OneatClassificationLoss
├── Data
│   ├── pytorch_datasets.py        # H5VisionDataset, H5MitosisDataset
│   └── oneat_prediction_dataset.py # OneatPredictionDataset (seg-guided inference)
├── Transforms
│   ├── oneat_transforms.py     # Microscopy-specific augmentations
│   ├── oneat_presets.py        # Light/Medium/Heavy presets
│   └── time_series_presets.py  # Cell fate transforms + presets (order-preserving)
├── Training
│   ├── lightning_trainer.py    # MitosisInception trainer class
│   ├── optimizers.py           # Adam, SGD, LARS, AdamW
│   └── schedulers.py           # Cosine, WarmCosine, Step
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
- [Lightning Modules](src/kapoorlabs_lightning/README_litmodules.md) - Module architecture details

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

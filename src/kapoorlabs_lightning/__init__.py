try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .lightning_trainer import (
    AutoLightningModel,
    AutoLightningTrain,
    LightningData,
    LightningModel,
    LightningTrain,
)
from .pytorch_datasets import MiniEcoset, MNISTDataModule, TFIODataset
from .pytorch_loggers import CustomNPZLogger
from .pytorch_models import __all__ as all_pytorch_models
from .pytorch_transforms import Transforms

__all__ = [
    "MiniEcoset",
    "TFIODataset",
    "MNISTDataModule",
    "LightningModel",
    "AutoLightningModel",
    "LightningTrain",
    "AutoLightningTrain",
    "LightningData",
    "Transforms",
    "CustomNPZLogger",
]
__all__.extend(all_pytorch_models)

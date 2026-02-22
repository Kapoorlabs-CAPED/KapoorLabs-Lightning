try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .base_module import (
    BaseModule
   
)
from .autoencoder_module import  AutoEncoderModule
from .classification_module import ClassificationModule

from .lightning_trainer import (
    MitosisInception,
)

LightningModel = ClassificationModule
AutoLightningModel = AutoEncoderModule

from .pytorch_datasets import (
    PointCloudDataset,
    PyntCloud,
    MitosisDataset,
    H5MitosisDataset,
    GenericDataset,
    GenericDataModule,
    H5VisionDataset,
)
from .pytorch_loggers import CustomNPZLogger
from .pytorch_models import __all__ as all_pytorch_models
from .pytorch_transforms import Transforms
from .time_series_transforms import (
    AddGaussianNoise as TimeSeriesAddGaussianNoise,
    RandomTimeShift,
    RandomScaling,
    RandomMasking,
    RandomTimePermutation,
    RandomTimeWarping,
    get_time_series_transforms,
)
from .oneat_transforms import (
    AddGaussianNoise as OneatAddGaussianNoise,
    AddPoissonNoise,
    SpatialGaussianBlur,
    RandomBrightnessContrast,
    RandomSpatialFlip,
    RandomSpatialRotation90,
    RandomIntensityScaling,
    PercentileNormalize,
    MinMaxNormalize as OneatMinMaxNormalize,
    ToFloat32,
    ElasticDeformation,
)
from .oneat_presets import (
    OneatTrainPresetLight,
    OneatTrainPresetMedium,
    OneatTrainPresetHeavy,
    OneatEvalPreset,
)
from .utils import get_most_recent_file, plot_npz_files, plot_npz_files_interactive, create_event_dataset_h5, percentile_norm, normalize_mi_ma
from .pytorch_callbacks import (
    CustomDeviceStatsMonitor,
    ExponentialDecayCallback,
    FineTuneLearningRateFinder,
    SaveFilesCallback,
    EarlyStoppingCall,
    CustomProgressBar,
    CheckpointModel,
    CustomVirtualMemory,
)

from .optimizers import Adam, RMSprop, Rprop, SGD, LARS, AdamWClipStyle

__all__ = [
    "BaseModule",
    "ClassificationModule",
    "AutoEncoderModule",
    "AutoLightningModel",
    "AutoLightningTrain",
    "CustomNPZLogger",
    "LightningModel",
    "LightningTrain",
    "PointCloudDataset",
    "PyntCloud",
    "Transforms",
    "get_most_recent_file",
    "CustomDeviceStatsMonitor",
    "ExponentialDecayCallback",
    "FineTuneLearningRateFinder",
    "SaveFilesCallback",
    "EarlyStoppingCall",
    "CustomProgressBar",
    "CheckpointModel",
    "CustomVirtualMemory",
    "MitosisDataset",
    "H5MitosisDataset",
    "Adam",
    "RMSprop",
    "Rprop",
    "SGD", "LARS", "AdamWClipStyle",
    "MitosisInception",
    "plot_npz_files",
    "plot_npz_files_interactive",
    "create_event_dataset_h5",
    "percentile_norm",
    "normalize_mi_ma",
    "OneatAddGaussianNoise",
    "AddPoissonNoise",
    "SpatialGaussianBlur",
    "RandomBrightnessContrast",
    "RandomSpatialFlip",
    "RandomSpatialRotation90",
    "RandomIntensityScaling",
    "PercentileNormalize",
    "OneatMinMaxNormalize",
    "ToFloat32",
    "ElasticDeformation",
    "OneatTrainPresetLight",
    "OneatTrainPresetMedium",
    "OneatTrainPresetHeavy",
    "OneatEvalPreset",
    "TimeSeriesAddGaussianNoise",
    "RandomTimeShift",
    "RandomScaling",
    "RandomMasking",
    "RandomTimePermutation",
    "RandomTimeWarping",
    "get_time_series_transforms",
    "H5VisionDataset",
    "GenericDataset",
    "GenericDataModule",
]
__all__.extend(all_pytorch_models)

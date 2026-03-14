from dataclasses import dataclass
from kapoorlabs_lightning.schedulers import _Schedulers


@dataclass
class CareParams:
    # UNet architecture
    unet_depth: int
    num_channels_init: int
    use_batch_norm: bool

    # Patch shape
    patch_z: int
    patch_y: int
    patch_x: int

    # Training parameters
    learning_rate: float
    batch_size: int
    epochs: int
    num_workers: int
    devices: int
    accelerator: str
    train_precision: str
    strategy: str
    gradient_clip_val: float
    gradient_clip_algorithm: str
    slurm_auto_requeue: bool
    alpha: float
    # Transform parameters
    transform_preset: str
    percentile_norm: bool
    pmin: float
    pmax: float
    gaussian_noise_std: float
    spatial_flip_p: float
    rotation_p: float

    # Optimizer
    weight_decay: float
    eta_min: float
    t_warmup: int

    # Prediction parameters
    n_tiles: list
    tile_overlap: float
    file_type: str

    # Scheduler
    scheduler: _Schedulers


@dataclass
class CareDataPaths:
    base_data_dir: str
    low_dir: str
    high_dir: str
    care_h5_file: str
    log_path: str
    experiment_name: str


@dataclass
class CareTrainClass:
    parameters: CareParams
    train_data_paths: CareDataPaths

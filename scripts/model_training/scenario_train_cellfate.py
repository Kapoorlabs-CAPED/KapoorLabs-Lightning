from dataclasses import dataclass
from typing import List
from kapoorlabs_lightning.schedulers import _Schedulers


@dataclass
class Params:
    # Model architecture parameters
    growth_rate: int
    block_config: List
    num_init_features: int
    bottleneck_size: int
    kernel_size: int
    attn_heads: int
    seq_len: int

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

    # Dataset parameters
    num_classes: int
    compute_class_weights: bool

    # Transform parameters
    transform_preset: str
    gaussian_noise_std: float
    min_scale: float
    max_scale: float
    max_mask_ratio: float

    # Optimizer and scheduler
    weight_decay: float
    momentum: float
    eta_min: float
    t_warmup: int
    gamma: float

    # Model choice
    model_choice: str

    # Scheduler
    scheduler: _Schedulers


@dataclass
class Train_Data_Paths:
    base_data_dir: str
    cellfate_h5_file: str
    log_path: str
    experiment_name: str


@dataclass
class CellFateClass:
    parameters: Params
    train_data_paths: Train_Data_Paths

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

    # Inference parameters
    batch_size: int
    num_workers: int
    devices: int
    accelerator: str
    num_classes: int

    # Model choice
    model_choice: str

    # Scheduler (unused but present in yaml)
    scheduler: _Schedulers


@dataclass
class Train_Data_Paths:
    base_data_dir: str
    cellfate_h5_file: str
    log_path: str
    experiment_name: str


@dataclass
class CellFatePredictClass:
    parameters: Params
    train_data_paths: Train_Data_Paths

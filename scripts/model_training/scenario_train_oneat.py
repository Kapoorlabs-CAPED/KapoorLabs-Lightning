from dataclasses import dataclass
from typing import List
from kapoorlabs_lightning.pytorch_models import DenseVollNet
from kapoorlabs_lightning.schedulers import _Schedulers

@dataclass
class Params:
        # Model architecture parameters
        startfilter: int
        start_kernel: int
        mid_kernel: int
        imagex: int
        imagey: int
        imagez: int
        depth: dict
        growth_rate: int
        pool_first: bool

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
        poisson_noise_p: float
        blur_p: float
        spatial_flip_p: float
        rotation_p: float
        brightness_contrast_p: float
        elastic_p: float
        percentile_norm: bool
        pmin: float
        pmax: float

        # Optimizer and scheduler
        optimizer_choice: str
        scheduler_choice: str
        weight_decay: float
        momentum: float
        eta_min: float
        t_warmup: int
        gamma: float

        # ONEAT specific parameters
        stage_number: int
        size_tminus: int
        size_tplus: int
        reduction: float
        alpha: float
        n_tiles: List
        event_threshold: List
        event_confidence: List
        file_type: str
        nms_space: int
        nms_time: int
        normalizeimage: bool
        event_name: List
        event_label: str
        event_position_name: List
        event_position_label: List
        categories_json: str
        cord_json: str
        oneat_model: DenseVollNet
        scheduler: _Schedulers


@dataclass
class Train_Data_Paths:
        base_data_dir: str
        oneat_timelapse_data_raw: str
        oneat_timelapse_data_csv: str
        oneat_timelapse_data_seg: str
        oneat_h5_file: str
        log_path: str
        experiment_name: str

            


@dataclass
class OneatClass:
    parameters: Params
    train_data_paths: Train_Data_Paths
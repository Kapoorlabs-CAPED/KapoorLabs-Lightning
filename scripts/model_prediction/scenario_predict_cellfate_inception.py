from dataclasses import dataclass
from typing import Optional


@dataclass
class Params:
    # Model architecture
    growth_rate: int
    block_config: list
    num_init_features: int
    bottleneck_size: int
    kernel_size: int
    attn_heads: int
    seq_len: int
    num_classes: int
    model_choice: str

    # Prediction
    tracklet_length: int
    features: str  # 'shape', 'dynamic', or 'shape_dynamic'
    track_id_column: str
    t_min: Optional[float]
    t_max: Optional[float]

    # Input mode: 'csv' or 'xml'
    input_mode: str
    accelerator: str

    # Optional knobs (have safe defaults; declared here so the
    # dataclass-backed Hydra config doesn't reject the new yaml keys
    # in strict-struct mode).
    time_window: Optional[list] = None
    transition_time_determination: bool = True
    transition_window_span: int = 50
    transition_window_stride: int = 25
    transition_refine_levels: int = 2
    transition_refine_factor: int = 2
    per_block_cm: bool = True
    cm_window_span: int = 50
    cm_window_stride: int = 25


@dataclass
class Experiment_Data_Paths:
    # CSV mode
    csv_file: str

    # XML mode
    xml_file: str
    seg_file: str
    mask_file: str
    calibration: str

    # Model
    checkpoint_path: str

    # Class mapping: '0:Basal,1:Radial,2:Goblet'
    class_map: str

    # Output
    output_dir: str
    output_prefix: str


@dataclass
class CellFatePredictInceptionClass:
    parameters: Params
    experiment_data_paths: Experiment_Data_Paths

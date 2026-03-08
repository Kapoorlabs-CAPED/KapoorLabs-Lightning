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


@dataclass
class Data_Paths:
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
    data_paths: Data_Paths

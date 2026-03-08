from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Params:
    # Tracklet parameters
    tracklet_length: int
    stride: int
    features: str  # 'shape', 'dynamic', or 'shape_dynamic'
    normalize: bool
    test_size: float
    seed: int

    # Input mode: 'csv' or 'xml'
    input_mode: str

    # Label
    label_column: str
    track_id_column: str

    # Filtering
    t_min: Optional[float]
    t_max: Optional[float]
    min_track_duration: Optional[int]


@dataclass
class Data_Paths:
    # CSV mode
    csv_file: str

    # XML mode
    xml_file: str
    seg_file: str
    mask_file: str
    calibration: str

    # Label mapping (comma-separated "Name:0,Name:1,..." or empty for auto)
    label_map: str

    # Output
    output_h5_file: str


@dataclass
class CellFateDataGenClass:
    parameters: Params
    data_paths: Data_Paths

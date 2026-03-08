from dataclasses import dataclass


@dataclass
class Params:
    # Shape feature computation
    num_points: int

    # Channels to process
    do_nuclei: bool
    do_membrane: bool

    # Channel transfer
    do_channel_transfer: bool
    transfer_source_channel: str
    transfer_target_channel: str

    # Output
    save_dataframe: bool
    normalize_dataframe: bool


@dataclass
class Data_Paths:
    # Base directory for the experiment
    base_directory: str

    # Timelapse names (without extensions)
    timelapse_nuclei_name: str
    timelapse_membrane_name: str

    # Segmentation image directories
    seg_nuclei_directory: str
    seg_membrane_directory: str

    # Mask / region of interest
    mask_directory: str

    # TrackMate XML directory
    tracking_directory: str

    # Output directory for master XMLs and DataFrames
    output_directory: str

    # Calibration override (empty = use XML calibration)
    calibration: str


@dataclass
class MasterTrackingClass:
    parameters: Params
    data_paths: Data_Paths

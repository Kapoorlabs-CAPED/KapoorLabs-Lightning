from dataclasses import dataclass
from typing import List


@dataclass
class AccuracyPaths:
    predictions_csv: str
    ground_truth_csv: str
    output_dir: str


@dataclass
class AccuracyParams:
    event_name: str
    match_threshold_space: float
    match_threshold_time: int
    prediction_class_column: str
    save_errors: bool


@dataclass
class AccuracyConfig:
    paths: AccuracyPaths
    params: AccuracyParams

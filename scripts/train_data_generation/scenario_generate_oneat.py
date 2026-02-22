from dataclasses import dataclass
from typing import List
from kapoorlabs_lightning.pytorch_models import DenseVollNet

@dataclass
class Params:
   
        
        startfilter: int
        start_kernel: int
        mid_kernel: int
        learning_rate: float
        batch_size: int
        epochs: int
        stage_number: int
        size_tminus: int
        size_tplus: int
        imagex: int
        imagey: int
        imagez: int
        depth: dict
        reduction: float
        n_tiles: List
        event_threshold: List
        event_confidence: List
        file_type: str
        nms_space: int
        nms_time: int
        normalizeimage: bool
        event_name: List
        event_label: str
        event_position_name : List
        event_position_label : List
        categories_json: str
        cord_json: str
        train_split: float
        batch_write_size: int
        oneat_model: DenseVollNet


@dataclass
class Train_Data_Paths:
      
        base_data_dir: str
        oneat_timelapse_data_raw : str
        oneat_timelapse_data_csv : str
        oneat_timelapse_data_seg : str
        oneat_h5_file: str

            


@dataclass
class OneatDataClass:
    parameters: Params
    train_data_paths: Train_Data_Paths
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from kapoorlabs_lightning.lightning_trainer import MitosisInception
from scenario_train_oneat import OneatClass
from kapoorlabs_lightning.pytorch_callbacks import (
    CheckpointModel,
    CustomProgressBar
)
from kapoorlabs_lightning.pytorch_loggers import CustomNPZLogger
from pytorch_lightning import seed_everything
from kapoorlabs_lightning.utils import save_config_as_json


configstore = ConfigStore.instance()
configstore.store(name="OneatClass", node=OneatClass)


@hydra.main(
    config_path="../conf", config_name="scenario_train_oneat"
)
def main(config: OneatClass):

    startfilter = config.parameters.startfilter
    start_kernel = config.parameters.start_kernel
    mid_kernel = config.parameters.mid_kernel
    learning_rate = config.parameters.learning_rate
    batch_size = config.parameters.batch_size
    epochs = config.parameters.epochs
    stage_number = config.parameters.stage_number
    size_tminus = config.parameters.size_tminus
    size_tplus = config.parameters.size_tplus
    imagex = config.parameters.imagex
    imagey = config.parameters.imagey
    imagez = config.parameters.imagez
    depth = config.parameters.depth
    reduction = config.parameters.reduction
    n_tiles = config.parameters.n_tiles
    event_threshold = config.parameters.event_threshold
    event_confidence = config.parameters.event_confidence
    file_type = config.parameters.file_type
    nms_space = config.parameters.nms_space
    nms_time = config.parameters.nms_time
    normalizeimage = config.parameters.normalizeimage
    event_name = config.parameters.event_name
    event_label = config.parameters.event_label
    event_position_name = config.parameters.event_position_name
    event_position_label = config.parameters.event_position_label
    categories_json = config.parameters.categories_json
    cord_json = config.parameters.cord_json
    oneat_model = hydra.utils.instantiate(config.parameters.oneat_model)
           




if __name__ == "__main__":
    main()    
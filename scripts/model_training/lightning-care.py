from pathlib import Path
import hydra
import os
from hydra.core.config_store import ConfigStore
from kapoorlabs_lightning.care_trainer import CareInception
from scenario_train_care import CareTrainClass
from kapoorlabs_lightning.pytorch_callbacks import (
    CheckpointModel,
    CustomProgressBar,
)
from kapoorlabs_lightning.pytorch_loggers import CustomNPZLogger
from kapoorlabs_lightning.utils import save_config_as_json


configstore = ConfigStore.instance()
configstore.store(name="CareTrainClass", node=CareTrainClass)


@hydra.main(config_path="../conf", config_name="scenario_train_care", version_base='1.3')
def main(config: CareTrainClass):
    # Extract parameters
    unet_depth = config.parameters.unet_depth
    num_channels_init = config.parameters.num_channels_init
    use_batch_norm = config.parameters.use_batch_norm

    learning_rate = config.parameters.learning_rate
    batch_size = config.parameters.batch_size
    epochs = config.parameters.epochs
    num_workers = config.parameters.num_workers
    devices = config.parameters.devices
    accelerator = config.parameters.accelerator
    train_precision = config.parameters.train_precision
    strategy = config.parameters.strategy
    gradient_clip_val = config.parameters.gradient_clip_val
    gradient_clip_algorithm = config.parameters.gradient_clip_algorithm
    slurm_auto_requeue = config.parameters.slurm_auto_requeue
    alpha = config.parameters.alpha
    weight_decay = config.parameters.weight_decay

    pmin = config.parameters.pmin
    pmax = config.parameters.pmax
    transform_preset = config.parameters.transform_preset
    gaussian_noise_std = config.parameters.gaussian_noise_std
    spatial_flip_p = config.parameters.spatial_flip_p
    rotation_p = config.parameters.rotation_p

    n_tiles = config.parameters.n_tiles
    tile_overlap = config.parameters.tile_overlap
    scheduler = hydra.utils.instantiate(
        config.parameters.scheduler, t_max=epochs, t_warmup=5, factor=alpha
    )

    # Data paths
    base_data_dir = config.train_data_paths.base_data_dir
    h5_file = os.path.join(base_data_dir, config.train_data_paths.care_h5_file)
    log_path = config.train_data_paths.log_path
    experiment_name = config.train_data_paths.experiment_name

    Path(log_path).mkdir(exist_ok=True, parents=True)
    save_config_as_json(config, log_path)

    progress = CustomProgressBar()
    modelcheckpoint = CheckpointModel(save_dir=log_path)

    callbacks = [
        progress.progress_bar,
        modelcheckpoint.checkpoint_callback,
    ]

    npz_logger = CustomNPZLogger(
        save_dir=log_path, experiment_name=experiment_name
    )
    logger = [npz_logger]

    trainer = CareInception(
        h5_file=h5_file,
        num_workers=num_workers,
        epochs=epochs,
        log_path=log_path,
        batch_size=batch_size,
        accelerator=accelerator,
        devices=devices,
        experiment_name=experiment_name,
        scheduler=scheduler,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        slurm_auto_requeue=slurm_auto_requeue,
        train_precision=train_precision,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        strategy=strategy,
        n_tiles=n_tiles,
        tile_overlap=tile_overlap,
    )

    if transform_preset == "light":
        trainer.setup_care_transforms_light(
            pmin=pmin,
            pmax=pmax,
            spatial_flip_p=spatial_flip_p,
        )
    elif transform_preset == "heavy":
        trainer.setup_care_transforms_heavy(
            pmin=pmin,
            pmax=pmax,
            spatial_flip_p=spatial_flip_p,
            rotation_p=rotation_p,
            gaussian_noise_std=gaussian_noise_std,
        )
    else:  # medium (default)
        trainer.setup_care_transforms_medium(
            pmin=pmin,
            pmax=pmax,
            spatial_flip_p=spatial_flip_p,
            rotation_p=rotation_p,
            gaussian_noise_std=gaussian_noise_std,
        )

    trainer.setup_care_h5_datasets()
    trainer.setup_care_unet_model(
        unet_depth=unet_depth,
        num_channels_init=num_channels_init,
        use_batch_norm=use_batch_norm,
    )
    trainer.setup_adam()
    trainer.setup_learning_rate_scheduler()
    trainer.setup_care_lightning_model()
    trainer.train(logger=logger, callbacks=callbacks)


if __name__ == "__main__":
    main()

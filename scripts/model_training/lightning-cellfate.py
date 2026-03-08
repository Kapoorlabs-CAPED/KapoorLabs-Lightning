#!/usr/bin/env python3
"""
Train a cell fate classification model on time series tracking data.

Usage:
    python lightning-cellfate.py
    python lightning-cellfate.py train_data_paths=cellfate_default
    python lightning-cellfate.py parameters.transform_preset=heavy
    python lightning-cellfate.py parameters.model_choice=densenet
"""

from pathlib import Path
import hydra
import os
from hydra.core.config_store import ConfigStore
from kapoorlabs_lightning.lightning_trainer import MitosisInception
from scenario_train_cellfate import CellFateClass
from kapoorlabs_lightning.pytorch_callbacks import (
    CheckpointModel,
    CustomProgressBar
)
from kapoorlabs_lightning.pytorch_loggers import CustomNPZLogger
from kapoorlabs_lightning.utils import save_config_as_json


configstore = ConfigStore.instance()
configstore.store(name="CellFateClass", node=CellFateClass)


@hydra.main(
    config_path="../conf", config_name="scenario_train_cellfate"
)
def main(config: CellFateClass):
    num_classes = config.parameters.num_classes
    num_workers = config.parameters.num_workers
    epochs = config.parameters.epochs
    batch_size = config.parameters.batch_size
    learning_rate = config.parameters.learning_rate
    devices = config.parameters.devices
    accelerator = config.parameters.accelerator
    train_precision = config.parameters.train_precision
    strategy = config.parameters.strategy
    gradient_clip_val = config.parameters.gradient_clip_val
    gradient_clip_algorithm = config.parameters.gradient_clip_algorithm
    slurm_auto_requeue = config.parameters.slurm_auto_requeue
    weight_decay = config.parameters.weight_decay
    momentum = config.parameters.momentum
    eta_min = config.parameters.eta_min
    t_warmup = config.parameters.t_warmup
    gamma = config.parameters.gamma
    scheduler = hydra.utils.instantiate(
        config.parameters.scheduler, t_max=epochs, t_warmup=t_warmup, factor=learning_rate
    )

    # Model architecture
    growth_rate = config.parameters.growth_rate
    block_config = tuple(config.parameters.block_config)
    num_init_features = config.parameters.num_init_features
    bottleneck_size = config.parameters.bottleneck_size
    kernel_size = config.parameters.kernel_size
    attn_heads = config.parameters.attn_heads
    seq_len = config.parameters.seq_len
    model_choice = config.parameters.model_choice

    # Transform parameters
    transform_preset = config.parameters.transform_preset
    gaussian_noise_std = config.parameters.gaussian_noise_std
    min_scale = config.parameters.min_scale
    max_scale = config.parameters.max_scale
    max_mask_ratio = config.parameters.max_mask_ratio

    # Paths
    log_path = config.train_data_paths.log_path
    experiment_name = config.train_data_paths.experiment_name
    base_data_dir = config.train_data_paths.base_data_dir
    h5_file = os.path.join(base_data_dir, config.train_data_paths.cellfate_h5_file)

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

    trainer = MitosisInception(
        h5_file=h5_file,
        num_classes=num_classes,
        num_workers=num_workers,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        devices=devices,
        accelerator=accelerator,
        log_path=log_path,
        train_precision=train_precision,
        strategy=strategy,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        slurm_auto_requeue=slurm_auto_requeue,
        weight_decay=weight_decay,
        momentum=momentum,
        eta_min=eta_min,
        t_warmup=t_warmup,
        gamma=gamma,
        experiment_name=experiment_name,
        scheduler=scheduler,
        growth_rate=growth_rate,
        block_config=block_config,
        num_init_features=num_init_features,
        bottleneck_size=bottleneck_size,
        kernel_size=kernel_size,
        attn_heads=attn_heads,
        seq_len=seq_len,
    )

    # Setup transforms (no temporal order changes)
    if transform_preset == "light":
        trainer.setup_cellfate_transforms_light(
            gaussian_noise_std=gaussian_noise_std,
            min_scale=min_scale,
            max_scale=max_scale,
        )
    elif transform_preset == "medium":
        trainer.setup_cellfate_transforms_medium(
            gaussian_noise_std=gaussian_noise_std,
            min_scale=min_scale,
            max_scale=max_scale,
            max_mask_ratio=max_mask_ratio,
        )
    elif transform_preset == "heavy":
        trainer.setup_cellfate_transforms_heavy(
            gaussian_noise_std=gaussian_noise_std,
            min_scale=min_scale,
            max_scale=max_scale,
            max_mask_ratio=max_mask_ratio,
        )

    # Setup dataset (H5 with train_arrays/train_labels/val_arrays/val_labels)
    trainer.setup_gbr_h5_datasets()

    # Setup model
    if model_choice == "inception":
        trainer.setup_inception_qkv_model()
    elif model_choice == "densenet":
        trainer.setup_densenet_model()
    elif model_choice == "mitosisnet":
        trainer.setup_mitosisnet_model()

    trainer.setup_adam()
    trainer.setup_learning_rate_scheduler()
    trainer.setup_cellfate_lightning_model()
    trainer.train(logger=logger, callbacks=callbacks)


if __name__ == "__main__":
    main()

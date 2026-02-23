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
from kapoorlabs_lightning.utils import save_config_as_json


configstore = ConfigStore.instance()
configstore.store(name="OneatClass", node=OneatClass)


@hydra.main(
    config_path="../conf", config_name="scenario_train_oneat"
)
def main(config: OneatClass):
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
    event_position_label = config.parameters.event_position_label
    alpha = config.parameters.alpha
    weight_decay = config.parameters.weight_decay
    momentum = config.parameters.momentum
    eta_min = config.parameters.eta_min
    t_warmup = config.parameters.t_warmup
    gamma = config.parameters.gamma
    scheduler = hydra.utils.instantiate(
        config.parameters.scheduler, t_max = epochs, t_warmup = 5, factor = alpha
    )

    startfilter = config.parameters.startfilter
    start_kernel = config.parameters.start_kernel
    mid_kernel = config.parameters.mid_kernel
    imagex = config.parameters.imagex
    imagey = config.parameters.imagey
    imagez = config.parameters.imagez
    size_tminus = config.parameters.size_tminus
    size_tplus = config.parameters.size_tplus
    depth = config.parameters.depth
    growth_rate = config.parameters.growth_rate
    pool_first = config.parameters.pool_first
    transform_preset = config.parameters.transform_preset
    gaussian_noise_std = config.parameters.gaussian_noise_std
    poisson_noise_p = config.parameters.poisson_noise_p
    blur_p = config.parameters.blur_p
    spatial_flip_p = config.parameters.spatial_flip_p
    rotation_p = config.parameters.rotation_p
    brightness_contrast_p = config.parameters.brightness_contrast_p
    elastic_p = config.parameters.elastic_p
    percentile_norm = config.parameters.percentile_norm
    pmin = config.parameters.pmin
    pmax = config.parameters.pmax
    compute_class_weights = config.parameters.compute_class_weights
    log_path = config.train_data_paths.log_path
    experiment_name = config.train_data_paths.experiment_name
    h5_file = config.train_data_paths.oneat_h5_file

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
        scheduler = scheduler
    )

    n_time = size_tminus + size_tplus + 1
    input_shape = (n_time, imagez, imagey, imagex)
    categories = num_classes
    box_vector = len(event_position_label) 

    trainer.setup_densenet_vision_model(
        input_shape=input_shape,
        categories=categories,
        box_vector=box_vector,
        start_kernel=start_kernel,
        mid_kernel=mid_kernel,
        startfilter=startfilter,
        depth=depth,
        growth_rate=growth_rate,
        pool_first=pool_first,
    )

    if transform_preset == "light":
        trainer.setup_oneat_transforms_light(
            gaussian_noise_std=gaussian_noise_std,
            spatial_flip_p=spatial_flip_p,
            percentile_norm=percentile_norm,
            pmin=pmin,
            pmax=pmax,
        )
    elif transform_preset == "medium":
        trainer.setup_oneat_transforms_medium(
            gaussian_noise_std=gaussian_noise_std,
            poisson_noise_p=poisson_noise_p,
            blur_p=blur_p,
            spatial_flip_p=spatial_flip_p,
            rotation_p=rotation_p,
            percentile_norm=percentile_norm,
            pmin=pmin,
            pmax=pmax,
        )
    elif transform_preset == "heavy":
        trainer.setup_oneat_transforms_heavy(
            gaussian_noise_std=gaussian_noise_std,
            poisson_noise_p=poisson_noise_p,
            blur_p=blur_p,
            spatial_flip_p=spatial_flip_p,
            rotation_p=rotation_p,
            brightness_contrast_p=brightness_contrast_p,
            elastic_p=elastic_p,
            percentile_norm=percentile_norm,
            pmin=pmin,
            pmax=pmax,
        )
    
      
    trainer.setup_oneat_transforms_eval(
            percentile_norm=percentile_norm,
            pmin=pmin,
            pmax=pmax,
        )
    trainer.setup_vision_h5_datasets(
        compute_class_weights=compute_class_weights,
    )
    trainer.setup_sgd()
    trainer.setup_learning_rate_scheduler()
    trainer.setup_oneat_lightning_model()
    trainer.train(logger = logger, callbacks=callbacks)


if __name__ == "__main__":
    main()

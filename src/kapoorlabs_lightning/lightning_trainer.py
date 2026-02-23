import os
import fcntl
import shutil
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from pathlib import Path
import torch
from lightning import Callback,  LightningModule, Trainer
from subprocess import call
from datetime import timedelta
import logging
from types import FrameType
from typing import Any, Callable, Dict, Iterable, List, Optional, Union, Set
from torch.utils.data import DataLoader
from .utils import (
    load_checkpoint_model,
)
from .pytorch_models import (
    DenseNet,
    MitosisNet,
    InceptionNet,
    DenseVollNet,
)
from .pytorch_losses import VolumeYoloLoss

from torchvision import transforms
from .time_series_transforms import (
    get_time_series_transforms,
    AddGaussianNoise,
    RandomTimeShift,
    RandomScaling,
    RandomMasking,
    RandomTimePermutation,
    RandomTimeWarping,
)
from .oneat_presets import (
    OneatTrainPresetLight,
    OneatTrainPresetMedium,
    OneatTrainPresetHeavy,
    OneatEvalPreset,
)
import signal
import threading
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.profilers import Profiler
from lightning.pytorch.strategies import Strategy
from .pytorch_datasets import H5MitosisDataset, H5VisionDataset, GenericDataModule
import json
from .base_module import BaseModule, _restore_schedulers
from .oneat_module import OneatActionModule
from .pytorch_loggers import CustomNPZLogger
from .pytorch_callbacks import CheckpointModel, CustomProgressBar
from .optimizers import Adam, RMSprop, LARS, SGD, AdamWClipStyle, AdamW
from lightning.pytorch.trainer.connectors.accelerator_connector import (
    _LITERAL_WARN,
    _PRECISION_INPUT,
)


class MitosisInception:


    def __init__(
        self,
        npz_file: str = None,
        h5_file: str = None,
        num_classes: int = 2,
        growth_rate: int = 32,
        block_config: tuple = (6, 12, 24, 16),
        num_init_features: int = 32,
        bottleneck_size: int = 4,
        kernel_size: int = 7,
        num_workers: int = 1,
        epochs: int = 1,
        log_path="log_path",
        batch_size: int = 1,
        accelerator="cuda",
        devices=1,
        experiment_name: str = "experiment_name",
        scheduler: str = None, 
        learning_rate: float = 0.001,
        eta_min: float = 1.0e-8,
        momentum: float = 0.9,
        decay: float = 1e-4,
        epsilon: float = 1.0,
        gamma: float = 0.94,
        slurm_auto_requeue: bool = False,
        train_precision: str = "32",
        gradient_clip_val: float = None,
        gradient_clip_algorithm: str = None,
        milestones: List = None,
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        t_warmup: int = 0,
        t_max: int = None,
        weight_decay: float = 1e-5,
        eps: float = 1e-1,
        n_pos: list = (8,),
        attention_dim: int = 64,
        strategy: str = "auto",
        attn_heads = 8,
        seq_len = 25
    ):
        self.npz_file = npz_file
        self.h5_file = h5_file
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        self.block_config = block_config
        self.num_init_features = num_init_features
        self.bottleneck_size = bottleneck_size
        self.kernel_size = kernel_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.t_max = t_max if t_max is not None else self.epochs
        self.n_pos = n_pos
        self.batch_size = batch_size
        self.log_path = log_path
        self.accelerator = accelerator
        self.devices = devices
        self.experiment_name = experiment_name
        self.learning_rate = learning_rate
        self.eta_min = eta_min
        self.slurm_auto_requeue = slurm_auto_requeue
        self.train_precision = train_precision
        self.scheduler = scheduler
        self.map_location = "cuda" if torch.cuda.is_available() else "cpu"
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay = decay
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.milestones = milestones
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.t_warmup = t_warmup
        self.strategy = strategy
        self.weight_decay = weight_decay
        self.eps = eps
        self.attention_dim = attention_dim
        self.attn_heads=attn_heads
        self.seq_len=seq_len
        self.ckpt_path = load_checkpoint_model(self.log_path)

    def setup_oneat_transforms_light(
        self,
        gaussian_noise_std=0.01,
        spatial_flip_p=0.3,
        percentile_norm=True,
        pmin=1.0,
        pmax=99.8,
    ):
        """Setup light ONEAT transforms for microscopy XYZT data."""
        self.train_transforms = OneatTrainPresetLight(
            gaussian_noise_std=gaussian_noise_std,
            spatial_flip_p=spatial_flip_p,
            percentile_norm=percentile_norm,
            pmin=pmin,
            pmax=pmax,
        )
        self.val_transforms = OneatEvalPreset(
            percentile_norm=percentile_norm,
            pmin=pmin,
            pmax=pmax,
        )
        print("ONEAT Light transforms set up")

    def setup_oneat_transforms_medium(
        self,
        gaussian_noise_std=0.02,
        poisson_noise_p=0.3,
        blur_p=0.3,
        spatial_flip_p=0.5,
        rotation_p=0.5,
        percentile_norm=True,
        pmin=1.0,
        pmax=99.8,
    ):
        """Setup medium ONEAT transforms for microscopy XYZT data."""
        self.train_transforms = OneatTrainPresetMedium(
            gaussian_noise_std=gaussian_noise_std,
            poisson_noise_p=poisson_noise_p,
            blur_p=blur_p,
            spatial_flip_p=spatial_flip_p,
            rotation_p=rotation_p,
            percentile_norm=percentile_norm,
            pmin=pmin,
            pmax=pmax,
        )
        self.val_transforms = OneatEvalPreset(
            percentile_norm=percentile_norm,
            pmin=pmin,
            pmax=pmax,
        )
        print("ONEAT Medium transforms set up")

    def setup_oneat_transforms_heavy(
        self,
        gaussian_noise_std=0.03,
        poisson_noise_p=0.5,
        blur_p=0.5,
        spatial_flip_p=0.7,
        rotation_p=0.7,
        brightness_contrast_p=0.3,
        elastic_p=0.3,
        percentile_norm=True,
        pmin=1.0,
        pmax=99.8,
    ):
        """Setup heavy ONEAT transforms for microscopy XYZT data."""
        self.train_transforms = OneatTrainPresetHeavy(
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
        self.val_transforms = OneatEvalPreset(
            percentile_norm=percentile_norm,
            pmin=pmin,
            pmax=pmax,
        )
        print("ONEAT Heavy transforms set up")

    def setup_oneat_transforms_eval(
        self,
        percentile_norm=True,
        pmin=1.0,
        pmax=99.8,
    ):
        """Setup evaluation-only ONEAT transforms (no augmentation)."""
        self.train_transforms = OneatEvalPreset(
            percentile_norm=percentile_norm,
            pmin=pmin,
            pmax=pmax,
        )
        self.val_transforms = OneatEvalPreset(
            percentile_norm=percentile_norm,
            pmin=pmin,
            pmax=pmax,
        )
        print("ONEAT Eval transforms set up")

    def setup_timeseries_transforms_light(
        self,
        mean=0.0,
        std=0.01,
        min_scale=0.98,
        max_scale=1.02,
    ):
        """Setup light time series transforms (minimal augmentation)."""
        self.train_transforms = transforms.Compose([
            AddGaussianNoise(mean=mean, std=std),
            RandomScaling(min_scale=min_scale, max_scale=max_scale),
        ])
        self.val_transforms = None
        print("Time series Light transforms set up")

    def setup_timeseries_transforms_medium(
        self,
        mean=0.0,
        std=0.02,
        min_scale=0.95,
        max_shift=1,
        max_scale=1.05,
        max_mask_ratio=0.1,
    ):
        """Setup medium time series transforms (balanced augmentation)."""
        self.train_transforms = get_time_series_transforms(
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_shift=max_shift,
            max_scale=max_scale,
            max_mask_ratio=max_mask_ratio,
        )
        self.val_transforms = None
        print("Time series Medium transforms set up")

    def setup_timeseries_transforms_heavy(
        self,
        mean=0.0,
        std=0.03,
        min_scale=0.9,
        max_shift=2,
        max_scale=1.1,
        max_mask_ratio=0.2,
        permutation_p=0.3,
        warping_p=0.3,
    ):
        """Setup heavy time series transforms (aggressive augmentation)."""
        self.train_transforms = transforms.Compose([
            AddGaussianNoise(mean=mean, std=std),
            RandomTimeShift(max_shift=max_shift),
            RandomScaling(min_scale=min_scale, max_scale=max_scale),
            RandomMasking(max_mask_ratio=max_mask_ratio),
            RandomTimePermutation(segment_size=3, p=permutation_p),
            RandomTimeWarping(sigma=0.3, p=warping_p),
        ])
        self.val_transforms = None
        print("Time series Heavy transforms set up")

    def setup_timeseries_transforms_eval(self):
        """Setup evaluation-only time series transforms (no augmentation)."""
        self.train_transforms = None
        self.val_transforms = None
        print("Time series Eval transforms set up (no augmentation)")



    def setup_gbr_h5_datasets(self):
        if self.h5_file is not None:
            train_arrays_key = "train_arrays"
            train_labels_key = "train_labels"

            val_arrays_key = "val_arrays"
            val_labels_key = "val_labels"

            self.dataset_train = H5MitosisDataset(
                self.h5_file,
                train_arrays_key,
                train_labels_key,
                num_classes = self.num_classes,
                transforms=self.train_transforms if hasattr(self, 'train_transforms') else None
            )

            self.dataset_val = H5MitosisDataset(
                self.h5_file,
                val_arrays_key,
                val_labels_key,
                num_classes = self.num_classes,
                transforms=self.val_transforms if hasattr(self, 'val_transforms') else None
            )

            self.input_channels = self.dataset_train.input_channels

            self.datamodule = GenericDataModule(
                dataset_train=self.dataset_train,
                dataset_val=self.dataset_val,
                batch_size_train=self.batch_size,
                batch_size_val=self.batch_size,
                num_workers_train=self.num_workers,
                num_workers_val=self.num_workers,
            )

            print('Data loaded')

    

    def setup_vision_h5_datasets(
        self,
        compute_class_weights=True,
    ):
            """
            Setup vision H5 datasets using pre-configured transforms.

            Note: Call one of the setup_oneat_transforms_* methods first to configure transforms.
            If no transforms are set, datasets will be created without transforms.
            """
            train_transform = self.train_transforms if hasattr(self, 'train_transforms') else None
            val_transform = self.val_transforms if hasattr(self, 'val_transforms') else None

            self.dataset_train = H5VisionDataset(
                h5_file=self.h5_file,
                split="train",
                transforms=train_transform,
                num_classes=self.num_classes,
                compute_class_weights=compute_class_weights,
            )

            self.dataset_val = H5VisionDataset(
                h5_file=self.h5_file,
                split="val",
                transforms=val_transform,
                num_classes=self.num_classes,
                compute_class_weights=False,
            )
            
            

            self.datamodule = GenericDataModule(
                dataset_train=self.dataset_train,
                dataset_val=self.dataset_val,
                batch_size_train=self.batch_size,
                batch_size_val=self.batch_size,
                num_workers_train=self.num_workers,
                num_workers_val=self.num_workers,
            )

            # Get input channels from the first dimension of image shape (T, Z, Y, X)
            self.input_channels = self.dataset_train.images_dataset.shape[1]

            if compute_class_weights:
                self.class_weights_dict = self.dataset_train.get_class_weights()
                print(f"Class weights: {self.class_weights_dict}")

            print("Vision H5 datasets loaded")


    def setup_densenet_model(self):
        self.model = DenseNet(
            self.input_channels,
            self.num_classes,
            growth_rate=self.growth_rate,
            block_config=self.block_config,
            num_init_features=self.num_init_features,
            bottleneck_size=self.bottleneck_size,
            kernel_size=self.kernel_size,
        )
        print(f"Training Mitosis Inception Model {self.model}")

    def setup_densenet_vision_model(
        self,
        input_shape,
        categories,
        box_vector,
        start_kernel,
        mid_kernel,
        startfilter,
        depth,
        growth_rate,
        pool_first = True
    ):

        print("Setting up model")
        self.model = DenseVollNet(
            input_shape,
            categories,
            box_vector,
            start_kernel=start_kernel,
            mid_kernel=mid_kernel,
            startfilter=startfilter,
            depth=depth,
            growth_rate = growth_rate,
            pool_first = pool_first
        )
        self.categories = categories
        self.box_vector = box_vector

        print(f"Training Vision Inception Model {self.model}")

   

    def setup_inception_qkv_model(self):
        self.model = InceptionNet(
            input_channels=self.input_channels,
            num_classes=self.num_classes,
            growth_rate=self.growth_rate,
            block_config=self.block_config,
            num_init_features=self.num_init_features,
            bottleneck_size=self.bottleneck_size,
            kernel_size=self.kernel_size,
            attn_heads=self.attn_heads,
            seq_len=self.seq_len,
        )
        print(f"Training Inception DenseNet with QKV Model {self.model}")

    
    def setup_mitosisnet_model(self):
        self.model = MitosisNet(
            self.input_channels,
            self.num_classes,
        )

    def setup_logger(self):
        self.npz_logger = CustomNPZLogger(
            save_dir=self.log_path, experiment_name=self.experiment_name
        )
        self.logger = self.npz_logger

    def setup_checkpoint(self):
        self.ckpt_path = load_checkpoint_model(self.log_path)
        self.modelcheckpoint = CheckpointModel(save_dir=self.log_path)

    def setup_adam(self):
        self.optimizer = Adam(lr=self.learning_rate)

    def setup_rmsprop(self):
        self.optimizer = RMSprop(
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.decay,
            eps=self.epsilon,
        )
    def setup_sgd(self):
        self.optimizer = SGD(lr=self.learning_rate, weight_decay=self.decay, momentum=self.momentum)

    def setup_lars(self):
        self.optimizer = LARS(lr=self.learning_rate, momentum=self.momentum, weight_decay=self.decay)    

    def setup_adam_clip(self):
        self.optimizer = AdamWClipStyle(lr=self.learning_rate)

    def setup_adamw(self):
        self.optimizer = AdamW(lr=self.learning_rate, weight_decay=self.decay)
    
        

    def setup_learning_rate_scheduler(self, weights_only=False):
        if self.ckpt_path is not None:
            checkpoint = torch.load(
                self.ckpt_path,
                map_location=self.map_location,
                weights_only=weights_only,
            )

            self.scheduler = _restore_schedulers(self.scheduler, checkpoint)
            try:
                if "_last_lr" in checkpoint["lr_schedulers"][0].keys():
                    self.learning_rate = checkpoint["lr_schedulers"][0][
                        "_last_lr"
                    ][0]
                    self.optimizer.lr = self.learning_rate
            except IndexError:
                pass

    def setup_oneat_lightning_model(self, oneat_accuracy=False):
        
        self.class_weights_dict = getattr(self.dataset_train, "class_weights_dict", None)

        if self.class_weights_dict is not None:
            self.class_weights = torch.tensor(
                list(self.class_weights_dict.values()), dtype=torch.float
            )
        else:
            self.class_weights = None
        self.loss = VolumeYoloLoss(
                categories=self.categories,
                box_vector=self.box_vector,
                device=self.map_location,
                class_weights_dict=self.class_weights_dict
            )

        self.progress = CustomProgressBar()
        self.lightning_model = OneatActionModule(
            self.model,
            self.loss,
            self.optimizer,
            scheduler=self.scheduler,
            num_classes=self.num_classes,
            oneat_accuracy=oneat_accuracy,
        )
        model_hyperparameters = {
            "input_channels": self.input_channels,
            "num_classes": self.num_classes,
            "learning_rate": self.learning_rate,
            "model_path": self.log_path,
            "model_name": self.experiment_name,
            "growth_rate": self.growth_rate,
            "block_config": list(self.block_config),
            "num_init_features": self.num_init_features,
            "bottleneck_size": self.bottleneck_size,
            "kernel_size": self.kernel_size,
        }
        with open(
            os.path.join(self.log_path, self.experiment_name + ".json"), "w"
        ) as json_file:
            json.dump(model_hyperparameters, json_file)

    def train(self, logger=[], callbacks=[]):

        print("Starting training")
        lightning_train = LightningModelTrain(
            self.datamodule.train_dataloader(),
            self.datamodule.val_dataloader(),
            self.lightning_model,
            epochs=self.epochs,
            accelerator=self.accelerator,
            ckpt_path=self.ckpt_path,
            default_root_dir=self.log_path,
            strategy=self.strategy,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm=self.gradient_clip_algorithm,
            callbacks=callbacks,
            logger=logger,
            devices=self.devices,
            precision=self.train_precision,
            slurm_auto_requeue=self.slurm_auto_requeue,
        )
        lightning_train.train_model()



class LightningModelTrain:
    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        model: LightningModule,
        callbacks: List[Callback] = None,
        logger: Logger = None,
        ckpt_path: str = None,
        min_epochs: int = 1,
        epochs: int = 10,
        accelerator: str = "cuda",
        devices: int = 1,
        num_nodes: int = 1,
        strategy: str = "auto",
        enable_checkpointing: bool = True,
        rank_zero_only: bool = False,
        log_every_n_steps: int = 20,
        default_root_dir: str = None,
        slurm_auto_requeue: bool = True,
        use_slurm: bool = True,
        precision: _PRECISION_INPUT = "32-true",
        deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.callbacks = callbacks
        self.logger = logger
        self.slurm_auto_requeue = slurm_auto_requeue
        self.ckpt_path = ckpt_path
        self.min_epochs = min_epochs
        self.epochs = epochs
        self.accelerator = accelerator
        self.devices = devices
        self.strategy = strategy
        self.num_nodes = num_nodes
        self.enable_checkpointing = enable_checkpointing
        self.rank_zero_only = rank_zero_only
        self.log_every_n_steps = log_every_n_steps
        self.default_root_dir = default_root_dir
        self.precision = precision
        self.deterministic = deterministic
        self.use_slurm = use_slurm
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.gradient_clip_val = gradient_clip_val

    def train_model(self):
        if self.ckpt_path is not None:
            if not os.path.isfile(self.ckpt_path):
                self.ckpt_path = None
        if self.use_slurm:
            if self.slurm_auto_requeue:
                plugins = [SLURMEnvironment(requeue_signal=signal.SIGTERM)]
            else:
                plugins = [SLURMEnvironment(auto_requeue=False)]
        else:
            plugins = []

        self.trainer = LightningTrainer(
            accelerator=self.accelerator,
            devices=self.devices,
            strategy=self.strategy,
            logger=self.logger,
            num_nodes=self.num_nodes,
            callbacks=self.callbacks,
            min_epochs=self.min_epochs,
            max_epochs=self.epochs,
            default_root_dir=self.default_root_dir,
            enable_checkpointing=self.enable_checkpointing,
            log_every_n_steps=self.log_every_n_steps,
            num_sanity_val_steps=0,
            deterministic=self.deterministic,
            precision=self.precision,
            plugins=plugins,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm=self.gradient_clip_algorithm,
        )

        self.trainer.fit(
            self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
            ckpt_path=self.ckpt_path,
        )

        self.trainer.validate(
            model=self.model,
            dataloaders=self.val_dataloader,
            ckpt_path=self.ckpt_path,
            verbose=True,
        )

    def callback_metrics(self):
        return self.trainer.callback_metrics


class LightningTrainer(Trainer):
    def __init__(
        self,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        num_nodes: int = 1,
        precision: _PRECISION_INPUT = "32-true",
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        fast_dev_run: Union[int, bool] = False,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: int = -1,
        min_steps: Optional[int] = None,
        max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
        limit_train_batches: Optional[Union[int, float]] = None,
        limit_val_batches: Optional[Union[int, float]] = None,
        limit_test_batches: Optional[Union[int, float]] = None,
        limit_predict_batches: Optional[Union[int, float]] = None,
        overfit_batches: Union[int, float] = 0.0,
        val_check_interval: Optional[Union[int, float]] = None,
        check_val_every_n_epoch: Optional[int] = 1,
        num_sanity_val_steps: Optional[int] = None,
        log_every_n_steps: Optional[int] = None,
        enable_checkpointing: Optional[bool] = None,
        enable_progress_bar: Optional[bool] = None,
        enable_model_summary: Optional[bool] = None,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
        benchmark: Optional[bool] = None,
        inference_mode: bool = True,
        use_distributed_sampler: bool = True,
        profiler: Optional[Union[Profiler, str]] = None,
        detect_anomaly: bool = False,
        barebones: bool = False,
        plugins=None,
        sync_batchnorm: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        default_root_dir: Optional[_PATH] = None,
    ):
        super().__init__(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            logger=logger,
            callbacks=callbacks,
            fast_dev_run=fast_dev_run,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_steps=max_steps,
            min_steps=min_steps,
            max_time=max_time,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches,
            overfit_batches=overfit_batches,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            num_sanity_val_steps=num_sanity_val_steps,
            log_every_n_steps=log_every_n_steps,
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            deterministic=deterministic,
            benchmark=benchmark,
            inference_mode=inference_mode,
            use_distributed_sampler=use_distributed_sampler,
            profiler=profiler,
            detect_anomaly=detect_anomaly,
            barebones=barebones,
            plugins=plugins,
            sync_batchnorm=sync_batchnorm,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            default_root_dir=default_root_dir,
        )




class LightningModelTrain:

    """
    Class for training PyTorch Lightning models.

    Args:
        datamodule (LightningDataModule): LightningDataModule instance for data handling.
        model (LightningModule): The PyTorch Lightning model to be trained.
        callbacks (List[Callback]): List of PyTorch Lightning callbacks.
        logger (Logger): Logger for recording training logs.
        ckpt_path (str): Path to the checkpoint file.
        min_epochs (int): Minimum number of epochs to train the model.
        epochs (int): Total number of epochs to train the model.
        accelerator (str): Accelerator type for distributed training (e.g., 'cpu', 'gpu', 'tpu').
        devices (int): Number of devices to use for training.
        num_nodes (int): Number of nodes to use for distributed training.
        strategy (str): Distributed training strategy.
        enable_checkpointing (bool): Whether to enable checkpointing during training.
        rank_zero_only (bool): Whether to log only for the master process.
        log_every_n_steps (int): Frequency of logging steps.
        default_root_dir (str): Default root directory for logs and checkpoints.
        slurm_auto_requeue (bool): Whether to automatically requeue jobs on SLURM.
        use_slurm (bool): Whether to use SLURM for distributed training.
        precision (Union[int, str]): Precision for training (e.g., '16', '32', '16-true', '16-false').
        deterministic (Union[bool, str]): Whether to use deterministic training (True/False).
        gradient_clip_val (Optional[Union[int, float]]): Value to clip gradients during training.
        gradient_clip_algorithm (Optional[str]): Algorithm to use for gradient clipping.

    Methods:
        train_model(): Train the PyTorch Lightning model.
        callback_metrics(): Get callback metrics from the trainer.

    Attributes:
        datamodule (LightningDataModule): LightningDataModule instance for data handling.
        model (LightningModule): The PyTorch Lightning model to be trained.
        callbacks (List[Callback]): List of PyTorch Lightning callbacks.
        logger (Logger): Logger for recording training logs.
        ckpt_path (str): Path to the checkpoint file.
        min_epochs (int): Minimum number of epochs to train the model.
        epochs (int): Total number of epochs to train the model.
        accelerator (str): Accelerator type for distributed training.
        devices (int): Number of devices to use for training.
        num_nodes (int): Number of nodes to use for distributed training.
        strategy (str): Distributed training strategy.
        enable_checkpointing (bool): Whether to enable checkpointing during training.
        rank_zero_only (bool): Whether to log only for the master process.
        log_every_n_steps (int): Frequency of logging steps.
        default_root_dir (str): Default root directory for logs and checkpoints.
        slurm_auto_requeue (bool): Whether to automatically requeue jobs on SLURM.
        use_slurm (bool): Whether to use SLURM for distributed training.
        precision (str): Precision for training.
        deterministic (Union[bool, str]): Whether to use deterministic training.
        gradient_clip_val (Optional[Union[int, float]]): Value to clip gradients during training.
        gradient_clip_algorithm (Optional[str]): Algorithm to use for gradient clipping.
        accumulate_grad_batches: Accumulate gradient batches

    Raises:
        AssertionError: If ckpt_path is specified but the file does not exist.
    """

    def __init__(
        self,
        datamodule: GenericDataModule = None,
        model: LightningModule = None,
        train_dataloaders=None,
        val_dataloaders=None,
        callbacks: List[Callback] = None,
        logger: Logger = None,
        ckpt_path: str = None,
        min_epochs: int = 1,
        epochs: int = 10,
        accelerator: str = "cuda",
        devices: int = 1,
        num_nodes: int = 1,
        strategy: str = "auto",
        enable_checkpointing: bool = True,
        rank_zero_only: bool = False,
        log_every_n_steps: int = 20,
        default_root_dir: str = None,
        slurm_auto_requeue: bool = True,
        use_slurm: bool = True,
        precision: _PRECISION_INPUT = "32-true",
        deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        reload_dataloaders_every_n_epochs=0,
        accumulate_grad_batches: int = 1,
    ):
        self._datamodule = datamodule

        self.train_dataloaders = train_dataloaders
        self.val_dataloaders = val_dataloaders
        self.model = model
        self.callbacks = callbacks
        self.logger = logger
        self.slurm_auto_requeue = slurm_auto_requeue
        self.ckpt_path = ckpt_path
        self.min_epochs = min_epochs
        self.epochs = epochs
        self.accelerator = accelerator
        self.devices = devices
        self.strategy = strategy
        self.num_nodes = num_nodes
        self.enable_checkpointing = enable_checkpointing
        self.rank_zero_only = rank_zero_only
        self.log_every_n_steps = log_every_n_steps
        self.default_root_dir = default_root_dir
        self.precision = precision
        self.deterministic = deterministic
        self.use_slurm = use_slurm
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.gradient_clip_val = gradient_clip_val
        self.reload_dataloaders_every_n_epochs = (
            reload_dataloaders_every_n_epochs
        )
        self.accumulate_grad_batches = accumulate_grad_batches

        self._setup()

    def get_datamodule(self):
        return self._datamodule

    # Setter for datamodules
    def set_datamodule(self, datamodule: GenericDataModule):
        self._datamodule = datamodule

    def _setup(self):
        if self.use_slurm:
            if self.slurm_auto_requeue:
                plugins = [SLURMEnvironment(requeue_signal=signal.SIGTERM)]
            else:
                plugins = [SLURMEnvironment(auto_requeue=False)]
        else:
            plugins = []

        self.trainer = LightningTrainer(
            accelerator=self.accelerator,
            devices=self.devices,
            strategy=self.strategy,
            logger=self.logger,
            num_nodes=self.num_nodes,
            callbacks=self.callbacks,
            min_epochs=self.min_epochs,
            max_epochs=self.epochs,
            default_root_dir=self.default_root_dir,
            enable_checkpointing=self.enable_checkpointing,
            log_every_n_steps=self.log_every_n_steps,
            num_sanity_val_steps=0,
            deterministic=self.deterministic,
            precision=self.precision,
            plugins=plugins,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm=self.gradient_clip_algorithm,
            reload_dataloaders_every_n_epochs=self.reload_dataloaders_every_n_epochs,
            accumulate_grad_batches=self.accumulate_grad_batches,
        )

    def train_model(self):
        if self.ckpt_path is not None:
            if not os.path.isfile(self.ckpt_path):
                self.ckpt_path = None

        if self.slurm_auto_requeue:
            self.trainer._signal_connector = _KlabSignalConnector(
                self.trainer, self.model
            )
            self.trainer._signal_connector.register_signal_handlers()
        if self._datamodule is not None:
            self.trainer.fit(
                self.model,
                datamodule=self.get_datamodule(),
                ckpt_path=self.ckpt_path,
            )

        elif (
            self._datamodule is None
            and self.train_dataloaders is not None
            and self.val_dataloaders is not None
        ):
            self.trainer.fit(
                self.model,
                train_dataloaders=self.train_dataloaders,
                val_dataloaders=self.val_dataloaders,
                ckpt_path=self.ckpt_path,
            )
        elif (
            self._datamodule is None
            and self.train_dataloaders is not None
            and self.val_dataloaders is None
        ):
            self.trainer.fit(
                self.model,
                train_dataloaders=self.train_dataloaders,
                ckpt_path=self.ckpt_path,
            )
        elif (
            self._datamodule is None
            and self.train_dataloaders is None
            and self.val_dataloaders is not None
        ):
            self.trainer.fit(
                self.model,
                val_dataloaders=self.val_dataloaders,
                ckpt_path=self.ckpt_path,
            )
        else:
            raise ValueError(
                "No datamodule or train or validation dataloaders provided"
            )

    def callback_metrics(self):
        return self.trainer.callback_metrics


class LightningTrainer(Trainer):
    """
    A PyTorch Lightning Trainer subclass for training Lightning models.

    Args:
        accelerator (Union[str, Accelerator]): Accelerator type for distributed training.
        strategy (Union[str, Strategy]): Distributed training strategy.
        devices (Union[List[int], str, int]): Device(s) to use for training.
        num_nodes (int): Number of nodes to use for distributed training.
        precision (_PRECISION_INPUT): Precision for training (e.g., '16', '32', '16-true', '16-false').
        logger (Optional[Union[Logger, Iterable[Logger], bool]]): Logger for recording training logs.
        callbacks (Optional[Union[List[Callback], Callback]]): Callbacks for monitoring/tracking training.
        fast_dev_run (Union[int, bool]): Whether to run a fast development mode.
        max_epochs (Optional[int]): Maximum number of epochs to train the model.
        min_epochs (Optional[int]): Minimum number of epochs to train the model.
        max_steps (int): Maximum number of training steps.
        min_steps (Optional[int]): Minimum number of training steps.
        max_time (Optional[Union[str, timedelta, Dict[str, int]]]): Maximum time for training.
        limit_train_batches (Optional[Union[int, float]]): Limiting training batches.
        limit_val_batches (Optional[Union[int, float]]): Limiting validation batches.
        limit_test_batches (Optional[Union[int, float]]): Limiting test batches.
        limit_predict_batches (Optional[Union[int, float]]): Limiting prediction batches.
        overfit_batches (Union[int, float]): Number of batches to use for overfitting.
        val_check_interval (Optional[Union[int, float]]): Validation check interval.
        check_val_every_n_epoch (Optional[int]): Check validation every n epochs.
        num_sanity_val_steps (Optional[int]): Number of sanity validation steps.
        log_every_n_steps (Optional[int]): Log frequency (in steps).
        enable_checkpointing (Optional[bool]): Whether to enable checkpointing.
        enable_progress_bar (Optional[bool]): Whether to enable progress bar.
        enable_model_summary (Optional[bool]): Whether to enable model summary.
        accumulate_grad_batches (int): Accumulate gradient batches.
        gradient_clip_val (Optional[Union[int, float]]): Value to clip gradients.
        gradient_clip_algorithm (Optional[str]): Algorithm for gradient clipping.
        deterministic (Optional[Union[bool, _LITERAL_WARN]]): Whether to use deterministic training.
        benchmark (Optional[bool]): Whether to use benchmark mode.
        inference_mode (bool): Whether to enable inference mode.
        use_distributed_sampler (bool): Whether to use distributed sampler.
        profiler (Optional[Union[Profiler, str]]): Profiler for profiling training.
        detect_anomaly (bool): Whether to detect anomalies during training.
        barebones (bool): Whether to use barebones mode.
        plugins: Additional plugins for trainer.
        sync_batchnorm (bool): Whether to synchronize batch normalization.
        reload_dataloaders_every_n_epochs (int): Reload dataloaders every n epochs.
        default_root_dir (Optional[_PATH]): Default root directory for logs and checkpoints.

    Attributes:
        All arguments passed to the constructor are available as attributes.

    Raises:
        NotImplementedError: If `accelerator` or `strategy` is set to 'auto'.
    """

    def __init__(
        self,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        num_nodes: int = 1,
        precision: _PRECISION_INPUT = "32-true",
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        fast_dev_run: Union[int, bool] = False,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: int = -1,
        min_steps: Optional[int] = None,
        max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
        limit_train_batches: Optional[Union[int, float]] = None,
        limit_val_batches: Optional[Union[int, float]] = None,
        limit_test_batches: Optional[Union[int, float]] = None,
        limit_predict_batches: Optional[Union[int, float]] = None,
        overfit_batches: Union[int, float] = 0.0,
        val_check_interval: Optional[Union[int, float]] = None,
        check_val_every_n_epoch: Optional[int] = 1,
        num_sanity_val_steps: Optional[int] = None,
        log_every_n_steps: Optional[int] = None,
        enable_checkpointing: Optional[bool] = None,
        enable_progress_bar: Optional[bool] = None,
        enable_model_summary: Optional[bool] = None,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
        benchmark: Optional[bool] = None,
        inference_mode: bool = True,
        use_distributed_sampler: bool = True,
        profiler: Optional[Union[Profiler, str]] = None,
        detect_anomaly: bool = False,
        barebones: bool = False,
        plugins=None,
        sync_batchnorm: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        default_root_dir: Optional[_PATH] = None,
    ):
        super().__init__(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            logger=logger,
            callbacks=callbacks,
            fast_dev_run=fast_dev_run,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_steps=max_steps,
            min_steps=min_steps,
            max_time=max_time,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches,
            overfit_batches=overfit_batches,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            num_sanity_val_steps=num_sanity_val_steps,
            log_every_n_steps=log_every_n_steps,
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            deterministic=deterministic,
            benchmark=benchmark,
            inference_mode=inference_mode,
            use_distributed_sampler=use_distributed_sampler,
            profiler=profiler,
            detect_anomaly=detect_anomaly,
            barebones=barebones,
            plugins=plugins,
            sync_batchnorm=sync_batchnorm,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            default_root_dir=default_root_dir,
        )


# copied from signal.pyi
_SIGNUM = Union[int, signal.Signals]
_HANDLER = Union[
    Callable[[_SIGNUM, FrameType], Any], int, signal.Handlers, None
]

log = logging.getLogger(__name__)


class _HandlersCompose:
    def __init__(
        self, signal_handlers: Union[List[_HANDLER], _HANDLER]
    ) -> None:
        if not isinstance(signal_handlers, list):
            signal_handlers = [signal_handlers]
        self.signal_handlers = signal_handlers

    def __call__(self, signum: _SIGNUM, frame: FrameType) -> None:
        for signal_handler in self.signal_handlers:
            if isinstance(signal_handler, int):
                signal_handler = signal.getsignal(signal_handler)
            if callable(signal_handler):
                signal_handler(signum, frame)


class _KlabSignalConnector:
    def __init__(self, trainer: LightningTrainer, model: BaseModule) -> None:
        self.received_sigterm = False
        self.trainer = trainer
        self.model = model
        self._original_handlers: Dict[_SIGNUM, _HANDLER] = {}

    def register_signal_handlers(self) -> None:
        self.received_sigterm = False
        self._original_handlers = self._get_current_signal_handlers()

        sigusr_handlers: List[_HANDLER] = []
        sigterm_handlers: List[_HANDLER] = [self._sigterm_notifier_fn]

        environment = self.trainer._accelerator_connector.cluster_environment
        if (
            isinstance(environment, SLURMEnvironment)
            and environment.auto_requeue
        ):
            log.info("SLURM auto-requeueing enabled. Setting signal handlers.")
            sigusr_handlers.append(self._slurm_sigusr_handler_fn)
            sigterm_handlers.append(self._sigterm_handler_fn)

        sigusr = (
            environment.requeue_signal
            if isinstance(environment, SLURMEnvironment)
            else signal.SIGUSR1
        )
        assert sigusr is not None
        if sigusr_handlers and not self._has_already_handler(sigusr):
            self._register_signal(sigusr, _HandlersCompose(sigusr_handlers))

        # we have our own handler, but include existing ones too
        if self._has_already_handler(signal.SIGTERM):
            sigterm_handlers.append(signal.getsignal(signal.SIGTERM))
        self._register_signal(
            signal.SIGTERM, _HandlersCompose(sigterm_handlers)
        )

    def _slurm_sigusr_handler_fn(self, signum: _SIGNUM, _: FrameType) -> None:
        rank_zero_info(f"Handling auto-requeue signal: {signum}")

        log.info("recieved sigusr, Klabs custom pytorch lightning handler")

        # save logger to make sure we get all the metrics
        for logger in self.trainer.loggers:
            logger.finalize("finished")
        # Save the metrics
        self._copy_files_on_sigterm()

    def _copy_files_on_sigterm(self) -> None:
        log.info("Copying files before handling SIGTERM.")
        present_files = os.listdir(self.trainer.default_root_dir)

        for file in present_files:
            if file.endswith(".npz") or file.endswith(".json"):
                backup_dir = os.path.join(
                    self.trainer.default_root_dir, "backup"
                )
                Path(backup_dir).mkdir(parents=True, exist_ok=True)

                with open(
                    os.path.join(self.trainer.default_root_dir, file), "rb"
                ) as src_file:
                    with open(
                        os.path.join(
                            backup_dir,
                            Path(file).stem
                            + f"_epoch_{self.trainer.current_epoch}_step_{self.trainer.global_step}"
                            + ".npz",
                        ),
                        "wb",
                    ) as dst_file:
                        fcntl.lockf(src_file.fileno(), fcntl.LOCK_SH)
                        shutil.copyfileobj(src_file, dst_file)
                        fcntl.lockf(src_file.fileno(), fcntl.LOCK_UN)

     

        if self.trainer.is_global_zero:
            # find job id
            array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
            if array_job_id is not None:
                array_task_id = os.environ["SLURM_ARRAY_TASK_ID"]
                job_id = f"{array_job_id}_{array_task_id}"
            else:
                job_id = os.environ["SLURM_JOB_ID"]

            cmd = ["scontrol", "requeue", job_id]

            # requeue job
            log.info(f"requeing job {job_id}...")
            try:
                result = call(cmd)
            except FileNotFoundError:
                # This can occur if a subprocess call to `scontrol` is run outside a shell context
                # Re-attempt call (now with shell context). If any error is raised, propagate to user.
                # When running a shell command, it should be passed as a single string.
                joint_cmd = [str(x) for x in cmd]
                result = call(" ".join(joint_cmd), shell=True)

            # print result text
            if result == 0:
                log.info(f"requeued exp {job_id}")
            else:
                log.warning("requeue failed...")

        input()

    def _sigterm_notifier_fn(self, signum: _SIGNUM, _: FrameType) -> None:
        log.info(
            rank_prefixed_message(
                f"Received SIGTERM: {signum}", self.trainer.local_rank
            )
        )
        # subprocesses killing the parent process is not supported, only the parent (rank 0) does it
        if not self.received_sigterm:
            # send the same signal to the subprocesses
            launcher = self.trainer.strategy.launcher
            if launcher is not None:
                launcher.kill(signum)
        self.received_sigterm = True

    def _sigterm_handler_fn(self, signum: _SIGNUM, _: FrameType) -> None:
        log.info(f"Bypassing SIGTERM: {signum}")

    def teardown(self) -> None:
        """Restores the signals that were previously configured before :class:`_SignalConnector` replaced them."""
        for signum, handler in self._original_handlers.items():
            if handler is not None:
                self._register_signal(signum, handler)
        self._original_handlers = {}

    @staticmethod
    def _get_current_signal_handlers() -> Dict[_SIGNUM, _HANDLER]:
        """Collects the currently assigned signal handlers."""
        valid_signals = _KlabSignalConnector._valid_signals()
        valid_signals -= {signal.SIGKILL, signal.SIGSTOP}
        return {signum: signal.getsignal(signum) for signum in valid_signals}

    @staticmethod
    def _valid_signals() -> Set[signal.Signals]:
        """Returns all valid signals supported on the current platform."""
        return signal.valid_signals()

    @staticmethod
    def _has_already_handler(signum: _SIGNUM) -> bool:
        return signal.getsignal(signum) not in (None, signal.SIG_DFL)

    @staticmethod
    def _register_signal(signum: _SIGNUM, handlers: _HANDLER) -> None:
        if threading.current_thread() is threading.main_thread():
            signal.signal(signum, handlers)  # type: ignore[arg-type]

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["_original_handlers"] = {}
        return state

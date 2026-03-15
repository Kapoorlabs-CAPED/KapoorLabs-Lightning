"""
CareInception: Orchestrator class for CARE denoising workflow.

Follows the MitosisInception pattern — setup methods called in sequence
to configure transforms, datasets, model, optimizer, scheduler, and
Lightning module before training.
"""

import json
import os

import torch

from .care_dataset import H5CareDataset
from .care_module import CareModule
from .care_presets import (
    CareTrainPresetLight,
    CareTrainPresetMedium,
    CareTrainPresetHeavy,
    CareEvalPreset,
)
from .base_module import _restore_schedulers
from .lightning_trainer import LightningModelTrain
from .optimizers import Adam, AdamW, SGD
from .pytorch_datasets import GenericDataModule
from .utils import load_checkpoint_model
from careamics.models.unet import UNet

class CareInception:
    """
    Orchestrator for CARE denoising training pipeline.

    Usage:
        trainer = CareInception(h5_file=..., ...)
        trainer.setup_care_transforms_medium(...)
        trainer.setup_care_h5_datasets()
        trainer.setup_care_unet_model(...)
        trainer.setup_adam()
        trainer.setup_learning_rate_scheduler()
        trainer.setup_care_lightning_model()
        trainer.train(logger=logger, callbacks=callbacks)
    """

    def __init__(
        self,
        h5_file: str = None,
        num_workers: int = 4,
        epochs: int = 100,
        log_path: str = "log_path",
        batch_size: int = 16,
        accelerator: str = "cuda",
        devices: int = 1,
        experiment_name: str = "care_denoising",
        scheduler=None,
        learning_rate: float = 0.0004,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        slurm_auto_requeue: bool = False,
        train_precision: str = "32-true",
        gradient_clip_val: float = None,
        gradient_clip_algorithm: str = None,
        strategy: str = "auto",
        n_tiles: list = None,
        tile_overlap: float = 0.125,
       
    ):
        self.h5_file = h5_file
        self.num_workers = num_workers
        self.epochs = epochs
        self.log_path = log_path
        self.batch_size = batch_size
        self.accelerator = accelerator
        self.devices = devices
        self.experiment_name = experiment_name
        self.scheduler = scheduler
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.slurm_auto_requeue = slurm_auto_requeue
        self.train_precision = train_precision
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.strategy = strategy
        self.n_tiles = n_tiles if n_tiles is not None else [1, 4, 4]
        self.tile_overlap = tile_overlap
       
        self.map_location = "cuda" if torch.cuda.is_available() else "cpu"
        self.ckpt_path = load_checkpoint_model(self.log_path)

    # ── Transform setup ──

    def setup_care_transforms_light(
        self,
        pmin=0.1,
        pmax=99.9,
        spatial_flip_p=0.5,
    ):
        self.train_transforms = CareTrainPresetLight(
            pmin=pmin,
            pmax=pmax,
            spatial_flip_p=spatial_flip_p,
        )
        self.val_transforms = CareEvalPreset(pmin=pmin, pmax=pmax)
        print("CARE Light transforms set up")

    def setup_care_transforms_medium(
        self,
        pmin=0.1,
        pmax=99.9,
        spatial_flip_p=0.5,
        rotation_p=0.5,
        gaussian_noise_std=0.01,
        gaussian_noise_p=0.3,
    ):
        self.train_transforms = CareTrainPresetMedium(
            pmin=pmin,
            pmax=pmax,
            spatial_flip_p=spatial_flip_p,
            rotation_p=rotation_p,
            gaussian_noise_std=gaussian_noise_std,
            gaussian_noise_p=gaussian_noise_p,
        )
        self.val_transforms = CareEvalPreset(pmin=pmin, pmax=pmax)
        print("CARE Medium transforms set up")

    def setup_care_transforms_heavy(
        self,
        pmin=0.1,
        pmax=99.9,
        spatial_flip_p=0.5,
        rotation_p=0.5,
        gaussian_noise_std=0.03,
        gaussian_noise_p=0.5,
    ):
        self.train_transforms = CareTrainPresetHeavy(
            pmin=pmin,
            pmax=pmax,
            spatial_flip_p=spatial_flip_p,
            rotation_p=rotation_p,
            gaussian_noise_std=gaussian_noise_std,
            gaussian_noise_p=gaussian_noise_p,
        )
        self.val_transforms = CareEvalPreset(pmin=pmin, pmax=pmax)
        print("CARE Heavy transforms set up")

    def setup_care_transforms_eval(self, pmin=0.1, pmax=99.9):
        self.train_transforms = CareEvalPreset(pmin=pmin, pmax=pmax)
        self.val_transforms = CareEvalPreset(pmin=pmin, pmax=pmax)
        print("CARE Eval transforms set up")

    # ── Dataset setup ──

    def setup_care_h5_datasets(self):
        train_transform = (
            self.train_transforms if hasattr(self, "train_transforms") else None
        )
        val_transform = (
            self.val_transforms if hasattr(self, "val_transforms") else None
        )

        self.dataset_train = H5CareDataset(
            self.h5_file, split="train", transforms=train_transform
        )
        self.dataset_val = H5CareDataset(
            self.h5_file, split="val", transforms=val_transform
        )

        self.datamodule = GenericDataModule(
            dataset_train=self.dataset_train,
            dataset_val=self.dataset_val,
            dataset_test=self.dataset_val,
            batch_size_train=self.batch_size,
            batch_size_val=self.batch_size,
            batch_size_test=self.batch_size,
            num_workers_train=self.num_workers,
            num_workers_val=self.num_workers,
            num_workers_test=self.num_workers,
        )

        print(
            f"CARE datasets loaded: {len(self.dataset_train)} train, "
            f"{len(self.dataset_val)} val patches "
            f"(shape: {self.dataset_train.patch_shape})"
        )

    # ── Model setup ──

    def setup_care_unet_model(
        self,
        unet_depth=3,
        num_channels_init=64,
        use_batch_norm=True,
        conv_dims=3,
        in_channels=1,
        num_classes=1,
    ):
        

        self.model = UNet(
            conv_dims=conv_dims,
            in_channels=in_channels,
            num_classes=num_classes,
            depth=unet_depth,
            num_channels_init=num_channels_init,
            use_batch_norm=use_batch_norm,
        )
        self.unet_depth = unet_depth
        self.num_channels_init = num_channels_init
        self.conv_dims = conv_dims 
        self.in_channels = in_channels 
        self.num_classes = num_classes

        print(
            f"CARE UNet: depth={unet_depth}, init_filters={num_channels_init}, \
            conv_dims={conv_dims}, in_channels={in_channels}, num_classes={num_classes}"
        )

    

    def setup_adam(self):
        self.optimizer = Adam(lr=self.learning_rate)

    def setup_adamw(self):
        self.optimizer = AdamW(
            lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def setup_sgd(self):
        self.optimizer = SGD(
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    

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


    def setup_care_lightning_model(self):
        self.loss = torch.nn.MSELoss()
        

        eval_transforms = getattr(self, "val_transforms", None)

        self.lightning_model = CareModule(
            self.model,
            self.loss,
            self.optimizer,
            scheduler=self.scheduler,
            n_tiles=self.n_tiles,
            tile_overlap=self.tile_overlap,
            eval_transforms=eval_transforms,
        )

        # Save hyperparameters
        model_hyperparameters = {
            "unet_depth": getattr(self, "unet_depth", 3),
            "num_channels_init": getattr(self, "num_channels_init", 64),
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "n_tiles": tuple(self.n_tiles),
            "tile_overlap": self.tile_overlap,
            "model_path": self.log_path,
            "model_name": self.experiment_name,
        }
        os.makedirs(self.log_path, exist_ok=True)
        with open(
            os.path.join(self.log_path, self.experiment_name + ".json"), "w"
        ) as json_file:
            json.dump(model_hyperparameters, json_file)

        print("CARE Lightning model set up")



    def train(self, logger=[], callbacks=[]):
        print("Starting CARE training")
        lightning_train = LightningModelTrain(
            self.datamodule,
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

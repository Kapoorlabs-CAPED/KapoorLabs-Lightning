import os
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from sklearn.cluster import KMeans
from torch import optim
import threading
from datetime import timedelta
import logging
import fcntl
import shutil
from subprocess import call
from types import FrameType
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning_utilities.core.rank_zero import rank_prefixed_message
from torch.nn import CosineSimilarity, BCEWithLogitsLoss, MSELoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from .utils import get_most_recent_file, load_checkpoint_model
from . import optimizers, schedulers
from .pytorch_models import (
    CloudAutoEncoder,
    DeepEmbeddedClustering,
    DenseNet,
    MitosisNet,
)
from .schedulers import (
    CosineAnnealingScheduler,
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
    WarmCosineAnnealingLR,
)
from torchmetrics.classification import Accuracy
import signal
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.profilers import Profiler
from lightning.pytorch.strategies import Strategy
from .pytorch_datasets import MitosisDataset
import json
from .pytorch_loggers import CustomNPZLogger
from .pytorch_callbacks import CheckpointModel, CustomProgressBar
from .optimizers import Adam, RMSprop
from lightning.pytorch.trainer.connectors.accelerator_connector import (
    _LITERAL_WARN,
    _PRECISION_INPUT,
)


class MitosisInception:

    LOSS_CHOICES = ["cross_entropy", "cosine", "bce", "mse"]
    SCHEDULER_CHOICES = ["cosine", "exponential", "multistep", "plateau", "warmup"]

    def __init__(
        self,
        npz_file: str,
        num_classes,
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
        loss_function: str = "cross_entropy",
        scheduler_choice: str = "cosine",
        experiment_name: str = "experiment_name",
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
        strategy: str = "auto",
    ):
        self.npz_file = npz_file
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        self.block_config = block_config
        self.num_init_features = num_init_features
        self.bottleneck_size = bottleneck_size
        self.kernel_size = kernel_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.t_max = t_max if t_max is not None else self.epochs

        self.batch_size = batch_size
        self.log_path = log_path
        self.accelerator = accelerator
        self.devices = devices
        self.experiment_name = experiment_name
        self.learning_rate = learning_rate
        self.eta_min = eta_min
        self.slurm_auto_requeue = slurm_auto_requeue
        self.train_precision = train_precision
        self.loss_function = loss_function
        self.scheduler_choice = scheduler_choice
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
        self.scheduler = None
        if self.loss_function not in self.LOSS_CHOICES:
            raise ValueError(
                f"Invalid loss function choice, must be one of {self.LOSS_CHOICES}"
            )
        if self.scheduler_choice not in self.SCHEDULER_CHOICES:
            raise ValueError(
                f"Invalid scheduler choice, must be one of {self.SCHEDULER_CHOICES}"
            )

    def setup_datasets(self):

        training_data = np.load(self.npz_file)
        train_dividing_arrays = training_data["dividing_train_arrays"]
        train_dividing_labels = training_data["dividing_train_labels"]
        train_non_dividing_arrays = training_data["non_dividing_train_arrays"]
        train_non_dividing_labels = training_data["non_dividing_train_labels"]

        val_dividing_arrays = training_data["dividing_val_arrays"]
        val_dividing_labels = training_data["dividing_val_labels"]
        val_non_dividing_arrays = training_data["non_dividing_val_arrays"]
        val_non_dividing_labels = training_data["non_dividing_val_labels"]
        print(
            f"Dividing labels in training {len(train_dividing_labels)}, Non Dividing labels in training {len(train_non_dividing_labels)}"
        )
        train_arrays = np.concatenate(
            (train_dividing_arrays, train_non_dividing_arrays)
        )
        train_labels = np.concatenate(
            (train_dividing_labels, train_non_dividing_labels)
        )

        self.dataset_train = MitosisDataset(train_arrays, train_labels)

        self.input_channels = self.dataset_train.input_channels

        val_arrays = np.concatenate((val_dividing_arrays, val_non_dividing_arrays))
        val_labels = np.concatenate((val_dividing_labels, val_non_dividing_labels))

        self.dataset_val = MitosisDataset(val_arrays, val_labels)

        self.mitosis_data = LightningData(
            data_train=self.dataset_train,
            data_val=self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        self.train_loader = self.mitosis_data.train_dataloader()
        self.val_loader = self.mitosis_data.val_dataloader()

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
        self.optimizer = Adam(
            lr=self.learning_rate
        )

    def setup_rmsprop(self):
        self.optimizer = RMSprop(
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.decay,
            eps=self.epsilon,
        )

    def setup_learning_rate_scheduler(self):
        if self.ckpt_path is not None:
            checkpoint = torch.load(self.ckpt_path, map_location=self.map_location)
            self.learning_rate = checkpoint["lr_schedulers"][0]["_last_lr"][0]
            self.optimizer.lr = self.learning_rate
        if self.scheduler_choice == "cosine":
            if self.ckpt_path is not None:
                self.t_max = checkpoint["lr_schedulers"][0]["T_max"]
                self.eta_min = checkpoint["lr_schedulers"][0]["eta_min"]
            self.scheduler = CosineAnnealingScheduler(
                t_max=self.t_max, eta_min=self.eta_min
            )
        if self.scheduler_choice == "exponential":
            self.scheduler = ExponentialLR(gamma=self.gamma)
        if self.scheduler_choice == "multistep":
            self.scheduler = MultiStepLR(milestones=self.milestones, gamma=self.gamma)
        if self.scheduler_choice == "plateau":
            self.scheduler = ReduceLROnPlateau(
                factor=self.factor, patience=self.patience, threshold=self.threshold
            )
        if self.scheduler_choice == "warmup":
            self.scheduler = WarmCosineAnnealingLR(
                t_warmup=self.t_warmup, t_max=self.t_max, eta_min=self.eta_min
            )

    def setup_lightning_model(self):
        if self.loss_function == "cross_entropy":
            self.loss = CrossEntropyLoss()
        if self.loss_function == "cosine":
            self.loss = CosineSimilarity()
        if self.loss_function == "bce":
            self.loss = BCEWithLogitsLoss()
        if self.loss_function == "mse":
            self.loss = MSELoss()

        self.progress = CustomProgressBar()
        self.lightning_model = LightningModel(
            self.model,
            self.loss,
            self.optimizer,
            scheduler=self.scheduler,
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

    def train(self):

        lightning_special_train = LightningModelTrain(
            self.train_loader,
            self.val_loader,
            self.lightning_model,
            epochs=self.epochs,
            accelerator=self.accelerator,
            ckpt_path=self.ckpt_path,
            default_root_dir=self.log_path,
            strategy=self.strategy,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm=self.gradient_clip_algorithm,
            callbacks=[
                self.progress.progress_bar,
                self.modelcheckpoint.checkpoint_callback,
            ],
            logger=self.logger,
            devices=self.devices,
            precision=self.train_precision,
            slurm_auto_requeue=self.slurm_auto_requeue,
        )
        lightning_special_train.train_model()


class LightningData(LightningDataModule):
    def __init__(
        self,
        data_train,
        data_val,
        batch_size=1,
        num_workers=0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_train = data_train
        self.data_val = data_val
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class LightningModel(LightningModule):
    def __init__(
        self,
        network: torch.nn.Module,
        loss_func: torch.nn.Module,
        optim_func: optim,
        scheduler: schedulers = None,
        automatic_optimization: bool = True,
        on_step: bool = True,
        on_epoch: bool = True,
        sync_dist: bool = True,
        rank_zero_only: bool = False,
        num_classes=2,
    ):
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["network", "loss_func", "optim_func", "scheduler"],
        )

        self.network = network
        self.loss_func = loss_func
        self.optim_func = optim_func
        self.scheduler = scheduler
        self.automatic_optimization = automatic_optimization
        self.on_step = on_step
        self.on_epoch = on_epoch
        self.sync_dist = sync_dist
        self.rank_zero_only = rank_zero_only
        self.num_classes = num_classes

    @classmethod
    def extract_json(cls, checkpoint_model_json):
        if checkpoint_model_json is not None:
            assert isinstance(
                checkpoint_model_json, (str, dict)
            ), "checkpoint_model_json must be a json or dict"

            if isinstance(checkpoint_model_json, str):
                with open(checkpoint_model_json) as file:
                    mitosis_data = json.load(file)
            else:
                mitosis_data = checkpoint_model_json
            return mitosis_data

    @classmethod
    def extract_mitosis_model(
        cls,
        mitosis_model,
        mitosis_model_json,
        loss_func,
        optim_func,
        scheduler=None,
        ckpt_model_path=None,
        map_location="cuda",
    ):
        if mitosis_model_json is not None:
            assert isinstance(
                mitosis_model_json, (str, dict)
            ), "checkpoint_model_json must be a json or dict"

            if isinstance(mitosis_model_json, str):
                with open(mitosis_model_json) as file:
                    mitosis_data = json.load(file)
            else:
                mitosis_data = mitosis_model_json

            num_classes = mitosis_data["num_classes"]
            input_channels = mitosis_data["input_channels"]
            if "growth_rate" in mitosis_data.keys():
                growth_rate = mitosis_data["growth_rate"]
                block_config = tuple(mitosis_data["block_config"])
                num_init_features = mitosis_data["num_init_features"]
                bottleneck_size = mitosis_data["bottleneck_size"]
                kernel_size = mitosis_data["kernel_size"]

            if ckpt_model_path is None:
                checkpoint_model_path = mitosis_data["model_path"]
                most_recent_checkpoint_ckpt = get_most_recent_file(
                    checkpoint_model_path, ".ckpt"
                )
            else:
                most_recent_checkpoint_ckpt = ckpt_model_path
            checkpoint = torch.load(
                most_recent_checkpoint_ckpt, map_location=map_location
            )
            learning_rate = 1.0e-3
            if isinstance(scheduler, CosineAnnealingScheduler):
                t_max = checkpoint["lr_schedulers"][0]["T_max"]
                eta_min = checkpoint["lr_schedulers"][0]["eta_min"]
                scheduler = scheduler(t_max=t_max, eta_min=eta_min)
                learning_rate = checkpoint["lr_schedulers"][0]["_last_lr"][0]
            if isinstance(scheduler, ExponentialLR):
                gamma = checkpoint["lr_schedulers"][0]["gamma"]
                scheduler = scheduler(gamma=gamma)
                learning_rate = checkpoint["lr_schedulers"][0]["_last_lr"][0]
            if isinstance(scheduler, MultiStepLR):
                milestones = checkpoint["lr_schedulers"][0]["milestones"]
                gamma = checkpoint["lr_schedulers"][0]["gamma"]
                scheduler = scheduler(milestones=milestones, gamma=gamma)
                learning_rate = checkpoint["lr_schedulers"][0]["_last_lr"][0]

            optimizer = optim_func(lr=learning_rate)
            network = mitosis_model(input_channels, num_classes)
            if "growth_rate" in mitosis_data.keys():
                network = mitosis_model(
                    input_channels,
                    num_classes,
                    growth_rate=growth_rate,
                    block_config=block_config,
                    num_init_features=num_init_features,
                    bottleneck_size=bottleneck_size,
                    kernel_size=kernel_size,
                )

            checkpoint_lightning_model = cls.load_from_checkpoint(
                most_recent_checkpoint_ckpt,
                network=network,
                loss_func=loss_func,
                optim_func=optimizer,
                scheduler=scheduler,
                map_location=map_location,
            )

            checkpoint_torch_model = checkpoint_lightning_model.network

            return checkpoint_lightning_model, checkpoint_torch_model

    def load_pretrained(self, pretrained_file, strict=True, verbose=True):
        if isinstance(pretrained_file, (list, tuple)):
            pretrained_file = pretrained_file[0]

        # Load the state dict
        state_dict = torch.load(pretrained_file)["state_dict"]

        # Make sure to have a weight dict
        if not isinstance(state_dict, dict):
            state_dict = dict(state_dict)

        # Get parameter dict of current model
        param_dict = dict(self.network.named_parameters())

        layers = []
        for layer in param_dict:
            if strict and not "network." + layer in state_dict:
                if verbose:
                    print(f'Could not find weights for layer "{layer}"')
                continue
            try:
                param_dict[layer].data.copy_(state_dict["network." + layer].data)
                layers.append(layer)
            except (RuntimeError, KeyError) as e:
                print(f"Error at layer {layer}:\n{e}")

        self.network.load_state_dict(param_dict)

        if verbose:
            print(f"Loaded weights for the following layers:\n{layers}")

    def forward(self, z):
        return self.network(z)

    def loss(self, y_hat, y):

        return self.loss_func(y_hat, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if not self.automatic_optimization:
            opt = self.optimizers()
            loss = self.loss(y_hat, y)
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
        else:
            loss = self.loss(y_hat, y)

        self.log(
            "train_loss",
            loss,
            on_step=self.on_step,
            on_epoch=self.on_epoch,
            prog_bar=True,
            logger=True,
            sync_dist=self.sync_dist,
            rank_zero_only=self.rank_zero_only,
        )

        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log(
            "learning_rate",
            current_lr,
            on_step=self.on_step,
            on_epoch=self.on_epoch,
            prog_bar=True,
            logger=True,
            sync_dist=self.sync_dist,
            rank_zero_only=self.rank_zero_only,
        )

        accuracy = self.compute_accuracy(y_hat, y)

        self.log(
            "train_accuracy",
            accuracy,
            on_step=self.on_step,
            on_epoch=self.on_epoch,
            prog_bar=True,
            logger=True,
            sync_dist=self.sync_dist,
            rank_zero_only=self.rank_zero_only,
        )

        return loss

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        accuracy = self.compute_accuracy(y_hat, y)
        self.log(
            f"{prefix}_loss",
            loss,
            on_epoch=self.on_epoch,
            prog_bar=True,
            logger=True,
            sync_dist=self.sync_dist,
            rank_zero_only=self.rank_zero_only,
        )

        self.log(
            f"{prefix}_accuracy",
            accuracy,
            on_epoch=self.on_epoch,
            prog_bar=True,
            logger=True,
            sync_dist=self.sync_dist,
            rank_zero_only=self.rank_zero_only,
        )

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "validation")

    def on_train_epoch_end(self) -> None:
        """Actions to perform at the end of each training epoch."""

        sch = self.lr_schedulers()
        learning_rate = self.optimizers().param_groups[0]["lr"]
        self.log(
            "learning_rate",
            learning_rate,
            prog_bar=True,
            logger=False,
            sync_dist=self.sync_dist,
            rank_zero_only=self.rank_zero_only,
        )
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics[self.reduce_lr_metric])

    def compute_accuracy(self, outputs, labels):

        predicted = outputs.data
        accuracy = Accuracy(task="multiclass", num_classes=self.num_classes).to(
            self.device
        )
        accuracies = accuracy(predicted, labels)

        return accuracies

    def configure_optimizers(self):
        optimizer = self.optim_func(self.parameters())

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            optimizer_scheduler = OrderedDict(
                {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "validation_loss",
                        "frequency": 1,
                    },
                }
            )
            return optimizer_scheduler
        return {"optimizer": optimizer}


class AutoLightningModel(LightningModule):
    def __init__(
        self,
        network: CloudAutoEncoder,
        loss_func: torch.nn.Module,
        optim_func: optim,
        scheduler: schedulers = None,
        scale_z=1,
        scale_xy=1,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["network", "loss_func", "optim_func", "scheduler"]
        )

        self.network = network
        self.loss_func = loss_func
        self.optim_func = optim_func
        self.scheduler = scheduler
        self.scale_z = scale_z
        self.scale_xy = scale_xy

    def forward(self, z):
        return self.network(z)

    def load_pretrained(self, pretrained_file, strict=True, verbose=True):
        if isinstance(pretrained_file, (list, tuple)):
            pretrained_file = pretrained_file[0]

        # Load the state dict
        state_dict = torch.load(pretrained_file)["state_dict"]
        # Make sure to have a weight dict
        if not isinstance(state_dict, dict):
            state_dict = dict(state_dict)
        # Get parameter dict of current model
        param_dict = dict(self.network.named_parameters())
        layers = []
        for layer in param_dict:
            if strict and not "network." + layer in state_dict:
                if verbose:
                    print(f'Could not find weights for layer "{layer}"')
                continue
            try:
                param_dict[layer].data.copy_(state_dict["network." + layer].data)
                layers.append(layer)
            except (RuntimeError, KeyError) as e:
                print(f"Error at layer {layer}:\n{e}")

        self.network.load_state_dict(param_dict)

        if verbose:
            (f"Loaded weights for the following layers:\n{layers}")

    def loss(self, y_hat, y):
        return self.loss_func(y_hat, y)

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]

        mean = torch.mean(batch, 1).to(self.device)
        scale = torch.tensor([[self.scale_z, self.scale_xy, self.scale_xy]]).to(
            self.device
        )

        outputs, features = self(batch)
        batch_size = batch.shape[0]
        outputs = outputs * scale
        for i in range(batch_size):
            outputs[i, :] = outputs[i, :] + mean[i, :]

        return outputs

    def training_step(self, batch, batch_idx):
        inputs = batch
        outputs, features = self(inputs)

        loss = self.loss_func(inputs, outputs)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        inputs = batch
        y_hat, features = self(inputs)
        self.loss(y_hat, inputs)

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "validation")

    def configure_optimizers(self):
        optimizer = self.optim_func(self.parameters())

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            optimizer_scheduler = OrderedDict(
                {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "validation_loss",
                        "frequency": 1,
                    },
                }
            )
            return optimizer_scheduler
        return {"optimizer": optimizer}


class ClusterLightningModel(LightningModule):
    def __init__(
        self,
        network: DeepEmbeddedClustering,
        loss_func: torch.nn.Module,
        cluster_loss_func: torch.nn.Module,
        optim_func: optim,
        train_dataloaders_inf,
        cluster_distribution,
        accelerator,
        devices,
        scheduler: schedulers = None,
        gamma: int = 1,
        update_interval: int = 1,
        divergence_tolerance: float = 1e-2,
        mem_percent: int = 40,
        q_power: int = 2,
        n_init: int = 20,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "network",
                "loss_func",
                "cluster_loss_func",
                "train_dataloaders_inf",
                "optim_func",
                "scheduler",
            ]
        )

        # params
        self.network = network
        self.loss_func = loss_func
        self.cluster_loss_func = cluster_loss_func
        self.optim_func = optim_func
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.devices = devices
        self.gamma = gamma
        self.train_dataloaders_inf = train_dataloaders_inf
        self.update_interval = update_interval
        self.divergence_tolerance = divergence_tolerance
        self.mem_percent = mem_percent
        self.count = 0
        self.cluster_distribution = cluster_distribution
        self.q_power = q_power
        self.n_init = n_init

    def load_pretrained(self, pretrained_file, strict=True, verbose=True):
        if isinstance(pretrained_file, (list, tuple)):
            pretrained_file = pretrained_file[0]
        # Load the state dict
        state_dict = torch.load(pretrained_file)["state_dict"]
        # Make sure to have a weight dict
        if not isinstance(state_dict, dict):
            state_dict = dict(state_dict)

        # Get parameter dict of current model
        param_dict = dict(self.network.named_parameters())

        layers = []
        for layer in param_dict:
            if strict and not "network." + layer in state_dict:
                if verbose:
                    print(f'Could not find weights for layer "{layer}"')
                continue
            try:
                param_dict[layer].data.copy_(state_dict["network." + layer].data)
                layers.append(layer)
            except (RuntimeError, KeyError) as e:
                print(f"Error at layer {layer}:\n{e}")

        self.network.load_state_dict(param_dict)

        if verbose:
            print(f"Loaded weights for the following layers:\n{layers}")

    def _get_target_distribution(self, out_distribution):
        numerator = (out_distribution**self.q_power) / torch.sum(
            out_distribution, axis=0
        )
        p = (numerator.t() / torch.sum(numerator, axis=1)).t()

        return p

    def forward(self, z):
        return self.network(z)

    def encoder_loss(self, y_hat, y):
        return self.loss_func(y_hat, y)

    def cluster_loss(self, clusters, tar_dist):
        return self.cluster_loss_func(F.log_softmax(clusters), F.softmax(tar_dist))

    def training_step(self, batch, batch_idx):
        self.compute_device = batch.device
        self.to(self.compute_device)
        self.target_distribution = self._get_target_distribution(
            self.cluster_distribution
        )
        batch_size = batch.shape[0]
        tar_dist = self.target_distribution[
            (batch_idx * batch_size) : ((batch_idx + 1) * batch_size),
            :,
        ]

        inputs = batch
        outputs, features, clusters = self(inputs)
        reconstruction_loss = self.loss_func(inputs, outputs)
        cluster_loss = self.cluster_loss(clusters, tar_dist.to(self.compute_device))
        loss = reconstruction_loss + self.gamma * cluster_loss

        tqdm_dict = {
            "reconstruction_loss": reconstruction_loss,
            "cluster_loss": cluster_loss,
            "epoch": self.current_epoch,
        }
        output = OrderedDict(
            {"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            rank_zero_only=True,
        )

        return output

    def on_train_epoch_end(self) -> None:
        if self.current_epoch > 0 and self.current_epoch % self.update_interval == 0:
            net, cluster_distribution = initialize_repeat_function(
                self.network,
                self.loss_func,
                self.cluster_loss_func,
                self.train_dataloaders_inf,
                self.optim_func,
                self.gamma,
                self.mem_percent,
                self.accelerator,
                self.devices,
                compute_device=self.compute_device,
                kmeans=False,
            )
            self.cluster_distribution = cluster_distribution.to(self.compute_device)
            self.to(self.compute_device)

    def configure_optimizers(self):
        optimizer = self.optim_func(self.parameters())

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            optimizer_scheduler = OrderedDict(
                {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "validation_loss",
                        "frequency": 1,
                    },
                }
            )
            return optimizer_scheduler
        return {"optimizer": optimizer}


class ClusterLightningDistModel(LightningModule):
    def __init__(
        self,
        network: DeepEmbeddedClustering,
        loss_func: torch.nn.Module,
        cluster_loss_func: torch.nn.Module,
        dataloader_inf: DataLoader,
        optim_func: optim,
        gamma: int = 1,
        update_interval: int = 5,
        divergence_tolerance: float = 1e-2,
        mem_percent: int = 40,
        q_power: int = 2,
        n_init: int = 20,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "network",
                "loss_func",
                "cluster_loss_func",
                "dataloader_inf",
                "optim_func",
            ]
        )

        self.network = network
        self.loss_func = loss_func
        self.cluster_loss_func = cluster_loss_func
        self.dataloader_inf = dataloader_inf
        self.optim_func = optim_func
        self.gamma = gamma
        self.update_interval = update_interval
        self.divergence_tolerance = divergence_tolerance
        self.mem_percent = mem_percent
        self.q_power = q_power
        self.n_init = n_init
        self.compute_device = self.device

    def load_pretrained(self, pretrained_file, strict=True, verbose=True):
        if isinstance(pretrained_file, (list, tuple)):
            pretrained_file = pretrained_file[0]
        # Load the state dict
        state_dict = torch.load(pretrained_file)["state_dict"]
        # Make sure to have a weight dict
        if not isinstance(state_dict, dict):
            state_dict = dict(state_dict)

        # Get parameter dict of current model
        param_dict = dict(self.network.named_parameters())

        layers = []
        for layer in param_dict:
            if strict and not "network." + layer in state_dict:
                if verbose:
                    print(f'Could not find weights for layer "{layer}"')
                continue
            try:
                param_dict[layer].data.copy_(state_dict["network." + layer].data)
                layers.append(layer)
            except (RuntimeError, KeyError) as e:
                print(f"Error at layer {layer}:\n{e}")

        self.network.load_state_dict(param_dict)

        if verbose:
            print(f"Loaded weights for the following layers:\n{layers}")

    def _initialise_centroid(self, results, kmeans=True):
        cluster_distribution = self._extract_features_distributions(results)
        if kmeans:
            km = KMeans(n_clusters=self.network.num_clusters, n_init=self.n_init)

            km.fit_predict(self.feature_array.detach().cpu().numpy())
            weights = torch.from_numpy(km.cluster_centers_)
            self.network.clustering_layer.set_weight(weights.to(self.compute_device))

        print("Cluster centres initialised")

        return self.network, cluster_distribution

    def _extract_features_distributions(self, results):
        cluster_distribution = None
        feature_array = None

        outputs, feature_array, cluster_distribution = zip(*results)
        self.feature_array = torch.stack(feature_array)[:, 0, :]
        self.cluster_distribution = torch.stack(cluster_distribution)[:, 0, :]

        self.feature_array = self.feature_array.to(self.compute_device)
        self.cluster_distribution = self.cluster_distribution.to(self.compute_device)
        self.predictions = torch.argmax(self.cluster_distribution.data, axis=1)
        self.predictions = self.predictions.to(self.compute_device)

        return self.cluster_distribution

    def forward(self, z):
        return self.network(z)

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        optimizer = self.optim_func(self.parameters())

        return {"optimizer": optimizer}


class LightningSpecialTrain:
    def __init__(
        self,
        datamodule: LightningDataModule,
        model: LightningModule,
        callbacks: List[Callback] = None,
        logger: Logger = None,
        ckpt_path: str = None,
        min_epochs: int = 1,
        epochs: int = 10,
        accelerator: str = "cpu",
        devices: int = 1,
        strategy: str = "auto",
        enable_checkpointing: bool = True,
    ):
        self.datamodule = datamodule
        self.model = model
        self.callbacks = callbacks
        self.logger = logger
        self.ckpt_path = ckpt_path
        self.min_epochs = min_epochs
        self.epochs = epochs
        self.accelerator = accelerator
        self.devices = devices
        self.strategy = strategy
        self.enable_checkpointing = enable_checkpointing

    def _train_model(self):
        if self.ckpt_path is None:
            self.default_root_dir = Path(self.ckpt_path).absolute().parent.as_posix()
        else:
            self.default_root_dir = os.getcwd()

        self.trainer = Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            strategy=self.strategy,
            logger=self.logger,
            callbacks=self.callbacks,
            min_epochs=self.min_epochs,
            max_epochs=self.epochs,
            default_root_dir=self.default_root_dir,
            enable_checkpointing=self.enable_checkpointing,
            precision=16,
        )

        self.trainer.fit(
            self.model,
            datamodule=self.datamodule,
            ckpt_path=self.ckpt_path,
        )

        self.trainer.validate(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=self.ckpt_path,
            verbose=True,
        )

    def callback_metrics(self):
        return self.trainer.callback_metrics


class LightningTrain:
    def __init__(
        self,
        dataset_train: Dataset,
        dataset_val: Dataset,
        loss_func: torch.nn.Module,
        model_func: torch.nn.Module,
        optim_func: optimizers._Optimizer,
        model_save_file: str,
        ckpt_file: str = None,
        train_val_test_split: List = [95, 2.5, 2.5],
        batch_size: int = 64,
        min_epochs: int = 1,
        epochs: int = 10,
        precision: int = 16,
        accelerator: str = "gpu",
        devices: int = -1,
        strategy: str = "auto",
        num_workers: int = 4,
        enable_checkpointing: bool = True,
        callbacks: List[Callback] = None,
        scheduler: schedulers = None,
        logger: Logger = None,
        **kwargs,
    ):
        self.dataset_train = dataset_train

        self.dataset_val = dataset_val

        self.loss_func = loss_func

        self.model_func = model_func

        self.optim_func = optim_func

        self.ckpt_file = ckpt_file

        self.model_save_file = model_save_file

        self.train_val_test_split = train_val_test_split

        self.batch_size = batch_size

        self.epochs = epochs

        self.accelerator = accelerator

        self.devices = devices

        self.enable_checkpointing = enable_checkpointing

        self.logger = logger

        self.callbacks = callbacks

        self.scheduler = scheduler

        self.strategy = strategy

        self.min_epochs = min_epochs

        self.precision = precision

        self.num_workers = num_workers

        self.hparams = {
            "loss_func": self.loss_func,
            "model_func": self.model_func,
            "min_epochs": self.min_epochs,
            "epochs": self.epochs,
            "optim_func": self.optim_func,
            "scheduler": self.scheduler,
            "train_val_test_split": self.train_val_test_split,
            "batch_size": self.batch_size,
            "dataset_train": self.dataset_train,
            "dataset_val": self.dataset_val,
        }
        self.hparams.update(kwargs=kwargs)

    def _train_model(self):
        self.model = LightningModel(
            self.model_func, self.loss_func, self.optim_func, self.scheduler
        )

        self.datas = LightningData(
            data_train=self.dataset_train,
            data_val=self.dataset_val,
            num_workers=self.num_workers,
        )
        self.default_root_dir = Path(self.model_save_file).absolute().parent.as_posix()
        self.default_root_dir = os.path.join(
            self.default_root_dir, Path(self.model_save_file).stem
        )
        Path(self.default_root_dir).mkdir(exist_ok=True)

        self.trainer = Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            strategy=self.strategy,
            logger=self.logger,
            callbacks=self.callbacks,
            min_epochs=self.min_epochs,
            max_epochs=self.epochs,
            default_root_dir=self.default_root_dir,
            enable_checkpointing=self.enable_checkpointing,
            precision=self.precision,
        )

        if self.ckpt_file is not None:
            self.trainer.fit(
                self.model,
                train_dataloaders=self.datas.train_dataloader(),
                val_dataloaders=self.datas.val_dataloader(),
                ckpt_path=self.ckpt_file,
            )

        else:
            self.trainer.fit(
                self.model,
                train_dataloaders=self.datas.train_dataloader(),
                val_dataloaders=self.datas.val_dataloader(),
            )

    def callback_metrics(self):
        return self.trainer.callback_metrics


class AutoLightningTrain:
    def __init__(
        self,
        dataset_train: Dataset,
        dataset_val: Dataset,
        loss_func: torch.nn.Module,
        model_func: torch.nn.Module,
        optim_func: optimizers._Optimizer,
        model_save_file: str,
        ckpt_file: str = None,
        train_val_test_split: List = [95, 2.5, 2.5],
        batch_size: int = 64,
        min_epochs: int = 1,
        epochs: int = 10,
        accelerator: str = "gpu",
        devices: int = -1,
        strategy: str = "auto",
        num_nodes: int = 1,
        num_workers: int = 4,
        enable_checkpointing: bool = True,
        callbacks: List[Callback] = None,
        scheduler: schedulers = None,
        logger: Logger = None,
        **kwargs,
    ):
        self.dataset_train = dataset_train

        self.dataset_val = dataset_val

        self.loss_func = loss_func

        self.model_func = model_func

        self.optim_func = optim_func

        self.ckpt_file = ckpt_file

        self.model_save_file = model_save_file

        self.train_val_test_split = train_val_test_split

        self.batch_size = batch_size

        self.epochs = epochs

        self.accelerator = accelerator

        self.devices = devices

        self.enable_checkpointing = enable_checkpointing

        self.logger = logger

        self.callbacks = callbacks

        self.scheduler = scheduler

        self.strategy = strategy

        self.min_epochs = min_epochs

        self.num_nodes = num_nodes

        self.num_workers = num_workers

        self.hparams = {
            "loss_func": self.loss_func,
            "model_func": self.model_func,
            "min_epochs": self.min_epochs,
            "epochs": self.epochs,
            "optim_func": self.optim_func,
            "scheduler": self.scheduler,
            "train_val_test_split": self.train_val_test_split,
            "batch_size": self.batch_size,
            "dataset_train": self.dataset_train,
            "dataset_val": self.dataset_val,
        }
        self.hparams.update(kwargs=kwargs)

    def _train_model(self):
        self.model = AutoLightningModel(
            self.model_func, self.loss_func, self.optim_func, self.scheduler
        )

        self.datas = LightningData(
            data_train=self.dataset_train,
            data_val=self.dataset_val,
            num_workers=self.num_workers,
        )
        self.default_root_dir = Path(self.model_save_file).absolute().parent.as_posix()
        self.default_root_dir = os.path.join(
            self.default_root_dir, Path(self.model_save_file).stem
        )
        Path(self.default_root_dir).mkdir(exist_ok=True)

        self.trainer = Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            strategy=self.strategy,
            logger=self.logger,
            callbacks=self.callbacks,
            min_epochs=self.min_epochs,
            max_epochs=self.epochs,
            default_root_dir=self.default_root_dir,
            enable_checkpointing=self.enable_checkpointing,
            num_nodes=self.num_nodes,
        )

        if self.ckpt_file is not None:
            self.trainer.fit(
                self.model,
                train_dataloaders=self.datas.train_dataloader(),
                val_dataloaders=self.datas.val_dataloader(),
                ckpt_path=self.ckpt_file,
            )

            self.trainer.validate(
                model=self.model,
                dataloaders=self.datas.val_dataloader(),
                ckpt_path=self.ckpt_file,
                verbose=True,
            )
        else:
            self.trainer.fit(
                self.model,
                train_dataloaders=self.datas.train_dataloader(),
                val_dataloaders=self.datas.val_dataloader(),
            )

            self.trainer.validate(
                model=self.model,
                dataloaders=self.datas.val_dataloader(),
                verbose=True,
            )

    def callback_metrics(self):
        return self.trainer.callback_metrics


class ClusterLightningTrain:
    def __init__(
        self,
        dataset_train: Dataset,
        dataset_val: Dataset,
        loss_func: torch.nn.Module,
        cluster_loss_func: torch.nn.Module,
        network: DeepEmbeddedClustering,
        optim_func: optimizers._Optimizer,
        model_save_file: str,
        ckpt_file: str = None,
        train_val_test_split: List = [95, 2.5, 2.5],
        gamma: int = 1,
        num_workers: int = 4,
        batch_size: int = 64,
        min_epochs: int = 1,
        epochs: int = 10,
        accelerator: str = "gpu",
        devices: int = -1,
        strategy: str = "auto",
        num_nodes: int = 1,
        enable_checkpointing: bool = True,
        callbacks: List[Callback] = None,
        scheduler: schedulers = None,
        logger: Logger = None,
        mem_percent: int = 20,
        **kwargs,
    ):
        self.dataset_train = dataset_train

        self.dataset_val = dataset_val

        self.loss_func = loss_func

        self.cluster_loss_func = cluster_loss_func

        self.gamma = gamma

        self.network = network

        self.optim_func = optim_func

        self.ckpt_file = ckpt_file

        self.model_save_file = model_save_file

        self.train_val_test_split = train_val_test_split

        self.batch_size = batch_size

        self.epochs = epochs

        self.accelerator = accelerator

        self.devices = devices

        self.enable_checkpointing = enable_checkpointing

        self.logger = logger

        self.callbacks = callbacks

        self.scheduler = scheduler

        self.strategy = strategy

        self.min_epochs = min_epochs

        self.num_nodes = num_nodes

        self.mem_percent = mem_percent

        self.num_workers = num_workers

        self.hparams = {
            "loss_func": self.loss_func,
            "network": self.network,
            "min_epochs": self.min_epochs,
            "epochs": self.epochs,
            "optim_func": self.optim_func,
            "scheduler": self.scheduler,
            "train_val_test_split": self.train_val_test_split,
            "batch_size": self.batch_size,
            "dataset_train": self.dataset_train,
            "dataset_val": self.dataset_val,
        }
        self.hparams.update(kwargs=kwargs)

    def _train_model(self):
        self.default_root_dir = Path(self.model_save_file).absolute().parent.as_posix()
        self.default_root_dir = os.path.join(
            self.default_root_dir, Path(self.model_save_file).stem
        )
        Path(self.default_root_dir).mkdir(exist_ok=True)
        self.trainer = Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            strategy=self.strategy,
            logger=self.logger,
            callbacks=self.callbacks,
            min_epochs=self.min_epochs,
            max_epochs=self.epochs,
            default_root_dir=self.default_root_dir,
            enable_checkpointing=self.enable_checkpointing,
            num_nodes=self.num_nodes,
        )

        self.datas = LightningData(
            data_train=self.dataset_train,
            data_val=self.dataset_val,
            num_workers=self.num_workers,
        )
        train_dataloaders = self.datas.train_dataloader()
        predict_loader = self.datas.predict_dataloader()

        if self.ckpt_file is None:
            get_kmeans = True
        else:
            get_kmeans = False
        net, cluster_distribution = initialize_repeat_function(
            self.network,
            self.loss_func,
            self.cluster_loss_func,
            predict_loader,
            self.optim_func,
            self.gamma,
            self.mem_percent,
            self.accelerator,
            self.devices,
            kmeans=get_kmeans,
        )

        self.model = ClusterLightningModel(
            net,
            self.loss_func,
            self.cluster_loss_func,
            self.optim_func,
            predict_loader,
            cluster_distribution,
            self.accelerator,
            self.devices,
            self.scheduler,
            gamma=self.gamma,
            mem_percent=self.mem_percent,
        )

        if self.ckpt_file is not None:
            self.trainer.fit(
                self.model,
                train_dataloaders=train_dataloaders,
                ckpt_path=self.ckpt_file,
            )

        else:
            self.trainer.fit(
                self.model,
                train_dataloaders=self.datas.train_dataloader(),
                val_dataloaders=self.datas.val_dataloader(),
            )

    def callback_metrics(self):
        return self.trainer.callback_metrics


def initialize_repeat_function(
    network,
    loss_func,
    cluster_loss_func,
    test_loaders,
    optim_func,
    gamma,
    mem_percent,
    accelerator,
    devices,
    compute_device=None,
    kmeans=False,
):
    premodel = ClusterLightningDistModel(
        network,
        loss_func,
        cluster_loss_func,
        test_loaders,
        optim_func,
        gamma=gamma,
        mem_percent=mem_percent,
    )
    if compute_device is not None:
        premodel.to(compute_device)

    pretrainer = Trainer(accelerator=accelerator, devices=devices)

    results = pretrainer.predict(model=premodel, dataloaders=test_loaders)

    net, cluster_distribution = premodel._initialise_centroid(results, kmeans=kmeans)
    if compute_device is not None:
        net.to(compute_device)
        cluster_distribution = cluster_distribution.to(compute_device)
    pretrainer._teardown()
    return net, cluster_distribution


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
        if self.slurm_auto_requeue:
            self.trainer._signal_connector = _KlabSignalConnector(
                self.trainer, self.model
            )
            self.trainer._signal_connector.register_signal_handlers()

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


# copied from signal.pyi
_SIGNUM = Union[int, signal.Signals]
_HANDLER = Union[Callable[[_SIGNUM, FrameType], Any], int, signal.Handlers, None]

log = logging.getLogger(__name__)


class _HandlersCompose:
    def __init__(self, signal_handlers: Union[List[_HANDLER], _HANDLER]) -> None:
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
    def __init__(self, trainer: LightningTrainer, model: LightningModel) -> None:
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
        if isinstance(environment, SLURMEnvironment) and environment.auto_requeue:
            log.info("SLURM auto-requeueing enabled. Setting signal handlers for H100.")
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
        self._register_signal(signal.SIGTERM, _HandlersCompose(sigterm_handlers))

    def _slurm_sigusr_handler_fn(self, signum: _SIGNUM, _: FrameType) -> None:
        rank_zero_info(f"Handling auto-requeue signal on H100: {signum}")

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
                backup_dir = os.path.join(self.trainer.default_root_dir, "backup")
                Path(backup_dir).mkdir(parents=True, exist_ok=True)

                # Lock the file before copying
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

        hpc_save_path = self.trainer._checkpoint_connector.hpc_save_path(
            self.trainer.default_root_dir
        )
        self.trainer.save_checkpoint(hpc_save_path)

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
        log.info(f"Bypassing SIGTERM on H100: {signum}")

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

import os
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from sklearn.cluster import KMeans
from torch import optim
from datetime import timedelta
import logging
from types import FrameType
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
from torch.nn import CosineSimilarity, BCEWithLogitsLoss, MSELoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from .utils import (
    get_most_recent_file,
    load_checkpoint_model,
    blockwise_causal_norm,
    blockwise_sum,
)
from . import optimizers, schedulers
from .pytorch_models import (
    CloudAutoEncoder,
    DeepEmbeddedClustering,
    DenseNet,
    MitosisNet,
    AttentionNet,
    HybridAttentionDenseNet,
    TrackAsuraTransformer,
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
from .pytorch_datasets import MitosisDataset, H5MitosisDataset
import json
from .pytorch_loggers import CustomNPZLogger
from .pytorch_callbacks import CheckpointModel, CustomProgressBar
from .optimizers import Adam, RMSprop
from lightning.pytorch.trainer.connectors.accelerator_connector import (
    _LITERAL_WARN,
    _PRECISION_INPUT,
)
from trackastra.data.wrfeat import get_features, build_windows
from trackastra.utils import normalize
from trackastra.model.predict import predict_windows


class MitosisInception:

    LOSS_CHOICES = ["cross_entropy", "cosine", "bce", "mse"]
    SCHEDULER_CHOICES = ["cosine", "exponential", "multistep", "plateau", "warmup"]

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
        n_pos: list=(8,),
        attention_dim: int = 64,
        strategy: str = "auto",
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
        self.attention_dim = attention_dim
        self.scheduler = None
        if self.loss_function not in self.LOSS_CHOICES:
            raise ValueError(
                f"Invalid loss function choice, must be one of {self.LOSS_CHOICES}"
            )
        if self.scheduler_choice not in self.SCHEDULER_CHOICES:
            raise ValueError(
                f"Invalid scheduler choice, must be one of {self.SCHEDULER_CHOICES}"
            )

    def setup_gbr_datasets(self):
        if self.npz_file is not None:
            training_data = np.load(self.npz_file)

            train_goblet_arrays = training_data["goblet_train_arrays"]
            train_goblet_labels = training_data["goblet_train_labels"]
            train_basal_arrays = training_data["basal_train_arrays"]
            train_basal_labels = training_data["basal_train_labels"]
            train_radial_arrays = training_data["radial_train_arrays"]
            train_radial_labels = training_data["radial_train_labels"]

            val_goblet_arrays = training_data["goblet_val_arrays"]
            val_goblet_labels = training_data["goblet_val_labels"]
            val_basal_arrays = training_data["basal_val_arrays"]
            val_basal_labels = training_data["basal_val_labels"]
            val_radial_arrays = training_data["radial_val_arrays"]
            val_radial_labels = training_data["radial_val_labels"]

            print(
                f"Goblet labels in training {len(train_goblet_labels)}, Radial labels in training {len(train_radial_labels)}, Basal labels in training {len(train_basal_labels)}"
            )
            train_arrays = np.concatenate(
                (train_goblet_arrays, train_basal_arrays, train_radial_arrays)
            )
            train_labels = np.concatenate(
                (train_goblet_labels, train_basal_labels, train_radial_labels)
            )

            self.dataset_train = MitosisDataset(train_arrays, train_labels)

            self.input_channels = self.dataset_train.input_channels

            val_arrays = np.concatenate(
                (val_goblet_arrays, val_basal_arrays, val_radial_arrays)
            )
            val_labels = np.concatenate(
                (val_goblet_labels, val_basal_labels, val_radial_labels)
            )

            self.dataset_val = MitosisDataset(val_arrays, val_labels)

            self.mitosis_data = LightningData(
                data_train=self.dataset_train,
                data_val=self.dataset_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )

            self.train_loader = self.mitosis_data.train_dataloader()
            self.val_loader = self.mitosis_data.val_dataloader()

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
            )

            self.dataset_val = H5MitosisDataset(
                self.h5_file,
                val_arrays_key,
                val_labels_key,
            )

            self.input_channels = self.dataset_train.input_channels

            self.mitosis_data = LightningData(
                data_train=self.dataset_train,
                data_val=self.dataset_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )

            self.train_loader = self.mitosis_data.train_dataloader()
            self.val_loader = self.mitosis_data.val_dataloader()

    def setup_h5_datasets(self):
        if self.h5_file is not None:
            train_arrays_key = "train_arrays"
            train_labels_key = "train_labels"
            val_arrays_key = "val_arrays"
            val_labels_key = "val_labels"

            self.dataset_train = H5MitosisDataset(
                self.h5_file, train_arrays_key, train_labels_key
            )

            self.dataset_val = H5MitosisDataset(
                self.h5_file, val_arrays_key, val_labels_key
            )

            self.input_channels = self.dataset_train.input_channels

            self.mitosis_data = LightningData(
                data_train=self.dataset_train,
                data_val=self.dataset_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )

            self.train_loader = self.mitosis_data.train_dataloader()
            self.val_loader = self.mitosis_data.val_dataloader()

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

    def setup_attention_model(self):

        self.model = AttentionNet(
            input_channels=self.input_channels,
            num_classes=self.num_classes,
            attention_dim=self.attention_dim,  # Add this as a parameter in your class
        )
        print(f"Training Attention Model {self.model}")

    def setup_hybrid_attention_model(self):
        self.model = HybridAttentionDenseNet(
            input_channels=self.input_channels,
            num_classes=self.num_classes,
            growth_rate=self.growth_rate,
            block_config=self.block_config,
            num_init_features=self.num_init_features,
            bottleneck_size=self.bottleneck_size,
            kernel_size=self.kernel_size,
            attention_dim=self.attention_dim,  
            n_pos = self.n_pos,
        )
        print(f"Training Hybrid DenseNet with Attention Model {self.model}")

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
            num_classes=self.num_classes,
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


logger = logging.getLogger(__name__)


class Trackasura(LightningModule):
    def __init__(
        self,
        network: TrackAsuraTransformer,
        optim_func: optim,
        causal_norm: bool = None,
        delta_cutoff: int = 2,
        tracking_frequency: int = -1,
        div_upweight: float = 20,
        scheduler: schedulers = None,
        automatic_optimization: bool = True,
        on_step: bool = True,
        on_epoch: bool = True,
        sync_dist: bool = True,
        rank_zero_only: bool = False,
    ):

        """
        Initializes the Trackasura model.

        Args:
            network (TrackAsuraTransformer): The neural network model.
            optim_func (optim): The optimizer function.
            causal_norm (bool, optional): Whether to apply causal normalization. Defaults to None.
            delta_cutoff (int, optional): Delta cutoff value. Defaults to 2.
            tracking_frequency (int, optional): Tracking frequency. Defaults to -1.
            div_upweight (float, optional): Upweight value. Defaults to 20.
            scheduler (schedulers, optional): The scheduler. Defaults to None.
            automatic_optimization (bool, optional): Whether to use automatic optimization. Defaults to True.
            on_step (bool, optional): Whether to apply on step. Defaults to True.
            on_epoch (bool, optional): Whether to apply on epoch. Defaults to True.
            sync_dist (bool, optional): Whether to synchronize distribution. Defaults to True.
            rank_zero_only (bool, optional): Whether to apply only to rank zero. Defaults to False.
        """

        super().__init__()

        self.save_hyperparameters(
            logger=False,
            ignore=["network", "optim_func", "scheduler"],
        )

        self.network = network
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.criterion_softmax = torch.nn.BCELoss(reduction="none")
        self.optim_func = optim_func
        self.casual_norm = causal_norm
        self.delta_cutoff = delta_cutoff
        self.tracking_frequency = tracking_frequency
        self.div_upweight = div_upweight
        self.scheduler = scheduler
        self.automatic_optimization = automatic_optimization
        self.on_step = on_step
        self.on_epoch = on_epoch
        self.sync_dist = sync_dist
        self.rank_zero_only = rank_zero_only

    @classmethod
    def extract_json(cls, checkpoint_model_json):

        """
        Extracts JSON checkpoint.

        Args:
            checkpoint_model_json (str, dict): Checkpoint model in JSON format.

        Returns:
            dict: Extracted JSON data.
        """

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
    def extract_trackastra_model(
        cls,
        trackastra_model,
        trackastra_model_json,
        optim_func,
        scheduler=None,
        ckpt_model_path=None,
        map_location="cuda",
    ):

        """
        Extracts Trackastra model.

        Args:
            trackastra_model: Trackastra model.
            trackastra_model_json (str, dict): Trackastra model in JSON format.
            optim_func: Optimizer function.
            scheduler: Scheduler. Defaults to None.
            ckpt_model_path: Checkpoint model path. Defaults to None.
            map_location (str, optional): Map location. Defaults to "cuda".

        Returns:
            tuple: Extracted Trackastra lightning model and Torch model.
        """

        if trackastra_model_json is not None:
            assert isinstance(
                trackastra_model_json, (str, dict)
            ), "checkpoint_model_json must be a json or dict"

            if isinstance(trackastra_model_json, str):
                with open(trackastra_model_json) as file:
                    trackastra_data = json.load(file)
            else:
                trackastra_data = trackastra_model_json

            coord_dim = trackastra_data["coord_dim"]
            embed_dim = trackastra_data["embed_dim"]
            n_head = trackastra_data["n_head"]
            cutoff_spatial = trackastra_data["cutoff_spatial"]
            cutoff_temporal = trackastra_data["cutoff_temporal"]
            n_spatial = trackastra_data["n_spatial"]
            n_temporal = trackastra_data["n_temporal"]
            dropout = trackastra_data["dropout"]
            mode = trackastra_data["mode"]

            if ckpt_model_path is None:
                checkpoint_model_path = trackastra_data["model_path"]
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
            network = trackastra_model(
                coord_dim=coord_dim,
                embed_dim=embed_dim,
                n_head=n_head,
                cutoff_spatial=cutoff_spatial,
                cutoff_temporal=cutoff_temporal,
                n_spatial=n_spatial,
                n_temposral=n_temporal,
                dropout=dropout,
                mode=mode,
            )

            checkpoint_lightning_model = cls.load_from_checkpoint(
                most_recent_checkpoint_ckpt,
                network=network,
                optim_func=optimizer,
                scheduler=scheduler,
                map_location=map_location,
            )

            checkpoint_torch_model = checkpoint_lightning_model.network

            return checkpoint_lightning_model, checkpoint_torch_model

    def load_pretrained(self, pretrained_file, strict=True, verbose=True):

        """
        Loads pretrained weights into the model.

        Args:
            pretrained_file (str or list): Path to the pretrained file or list of paths.
            strict (bool, optional): Whether to strictly load the weights. Defaults to True.
            verbose (bool, optional): Whether to print verbose messages. Defaults to True.
        """

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

    def _shared_eval(self, batch, batch_idx):

        """
        Shared evaluation function for training and validation steps.

        Args:
            batch (dict): Batch data.
            batch_idx (int): Batch index.

        Returns:
            dict: Evaluation results.
        """

        feats, coords, A, timepoints, padding_mask = (
            batch["features"],
            batch["coords"],
            batch["assoc_matrix"],
            batch["timepoints"],
            batch["padding_mask"].bool(),
        )

        A_pred = self.model(coords, feats, padding_mask=padding_mask)
        A_pred.clamp_(torch.finfo(torch.float16).min, torch.finfo(torch.float16).max)
        mask_invalid = torch.logical_or(
            padding_mask.unsqueeze(1), padding_mask.unsqueeze(2)
        )
        A_pred[mask_invalid] = 0
        loss = self.criterion(A_pred, A)

        if self.causal_norm is not None:
            A_pred_soft = torch.stack(
                [
                    blockwise_causal_norm(
                        _A, _t, mode=self.causal_norm, mask_invalid=_m
                    )
                    for _A, _t, _m in zip(A_pred, timepoints, mask_invalid)
                ]
            )
            loss = 0.01 * loss + self.criterion_softmax(A_pred_soft, A)

        with torch.no_grad():
            block_sum1 = torch.stack(
                [blockwise_sum(A, t, dim=-1) for A, t in zip(A, timepoints)], 0
            )
            block_sum2 = torch.stack(
                [blockwise_sum(A, t, dim=-2) for A, t in zip(A, timepoints)], 0
            )
            block_sum = A * (block_sum1 + block_sum2)

            normal_tracks = block_sum == 2
            division_tracks = block_sum > 2

            loss_weight = 1 + 1.0 * normal_tracks + self.div_upweight * division_tracks

        loss *= loss_weight

        mask_valid = ~mask_invalid
        dt = timepoints.unsqueeze(1) - timepoints.unsqueeze(2)
        mask_time = torch.logical_and(dt > 0, dt <= self.delta_cutoff)
        mask = mask_time * mask_valid
        mask = mask.float()

        loss_before_reduce = loss * mask
        loss_normalized = loss_before_reduce / (
            mask.sum(dim=(1, 2), keepdim=True) + torch.finfo(torch.float32).eps
        )
        loss_per_sample = loss_normalized.sum(dim=(1, 2))

        prefactor = torch.pow(mask.sum(dim=(1, 2)), 0.2)
        loss = (
            loss_per_sample
            * prefactor
            / (prefactor.sum() + torch.finfo(torch.float32).eps)
        ).sum()

        return {
            "loss": loss,
            "padding_fraction": padding_mask.float().mean(),
            "loss_before_reduce": loss_before_reduce,
            "A_pred": A_pred,
            "mask": mask,
            "mask_time": mask_time,
            "mask_valid": mask_valid,
        }

    def train_step(self, batch, batch_idx):

        out = self._shared_eval(batch)
        loss = out["loss"]
        if torch.isnan(loss):
            print("NaN loss, skipping")
            return None

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=self.on_step,
            on_epoch=self.on_epoch,
            sync_dist=self.sync_dist,
        )

        self.log_dict(
            {
                "detections_per_sequence": batch["coords"].shape[1],
                "padding_fraction": out["padding_fraction"],
            },
            on_step=True,
            on_epoch=False,
        )

        return loss

    def validation_step(self, batch, batch_idx):

        out = self._shared_eval(batch)
        loss = out["loss"]
        if torch.isnan(loss):
            print("NaN loss, skipping")
            return None

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=self.on_step,
            on_epoch=self.on_epoch,
            sync_dist=self.sync_dist,
        )

        return loss

    def predict_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        edge_threshold: float = 0.05,
        n_workers: int = 0,
    ):
        imgs, masks = batch

        logger.info("Predicting weights for candidate graph")

        imgs = normalize(imgs)

        self.network.eval()

        features = get_features(
            detections=masks,
            imgs=imgs,
            ndim=self.network.config["coord_dim"],
            n_workers=n_workers,
            progbar_class=tqdm,
        )

        logger.info("Building windows")

        windows = build_windows(
            features,
            window_size=self.network.config["window"],
            progbar_class=tqdm,
        )

        logger.info("Predicting windows")

        predictions = predict_windows(
            windows=windows,
            features=features,
            model=self.network,
            edge_threshold=edge_threshold,
            spatial_dim=masks.ndim - 1,
            progbar_class=tqdm,
        )

        return predictions

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
        local_model_path=None,
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
            if "attention_dim" in mitosis_data.keys():
                attention_dim = mitosis_data["attention_dim"]

            if ckpt_model_path is None:
                if local_model_path is None:
                    checkpoint_model_path = mitosis_data["model_path"]
                else:
                    checkpoint_model_path = local_model_path
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
            if (
                "attention_dim" in mitosis_data.keys()
                and "growth_rate" in mitosis_data.keys()
            ):
                network = mitosis_model(
                    input_channels,
                    num_classes,
                    growth_rate=growth_rate,
                    block_config=block_config,
                    num_init_features=num_init_features,
                    bottleneck_size=bottleneck_size,
                    kernel_size=kernel_size,
                    attention_dim=attention_dim,
                )
            if (
                "attention_dim" in mitosis_data.keys()
                and "growth_rate" not in mitosis_data.keys()
            ):

                network = mitosis_model(
                    input_channels, num_classes, attention_dim=attention_dim
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

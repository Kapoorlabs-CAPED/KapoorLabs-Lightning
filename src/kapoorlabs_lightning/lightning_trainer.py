import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List

import torch
from cellshape_cloud import CloudAutoEncoder
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers.logger import Logger
from sklearn.cluster import KMeans
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split

from . import optimizers, schedulers
from .pytorch_models import DeepEmbeddedClustering

sys.setrecursionlimit(100000)


class LightningData(LightningDataModule):
    def __init__(
        self,
        hparams,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_val_test_split: float = hparams["train_val_test_split"]
        self._batch_size: int = hparams["batch_size"]
        self.dataset: Dataset = hparams["dataset"]

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    def setup(self, stage: str):
        self.train_val_test_split = [
            int(self.train_val_test_split[i] * len(self.dataset) / 100)
            for i in range(len(self.train_val_test_split) - 1)
        ]
        self.train_val_test_split.append(
            len(self.dataset) - sum(self.train_val_test_split)
        )
        print(
            f"LightningData.setup() called with stage={stage} on dataset of length {len(self.dataset)} with {self.train_val_test_split} split."
        )
        self.data_train, self.data_val, self.data_test = random_split(
            dataset=self.dataset,
            lengths=self.train_val_test_split,
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self._batch_size,
            num_workers=os.cpu_count() // 3,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self._batch_size,
            num_workers=os.cpu_count() // 3,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self._batch_size,
            num_workers=os.cpu_count() // 3,
        )

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=1)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass


class LightningModel(LightningModule):
    def __init__(
        self,
        network: torch.nn.Module,
        loss_func: torch.nn.Module,
        optim_func: optim,
        scheduler: schedulers = None,
        automatic_optimization: bool = True,
        scalar: torch.cuda.amp.grad_scaler = None,
        use_blt_loss: bool = False,
        timesteps: int = 0,
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
        self.scalar = scalar
        self.use_blt_loss = use_blt_loss
        self.timesteps = timesteps

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
                param_dict[layer].data.copy_(
                    state_dict["network." + layer].data
                )
                layers.append(layer)
            except (RuntimeError, KeyError) as e:
                print(f"Error at layer {layer}:\n{e}")

        self.network.load_state_dict(param_dict)

        if verbose:
            print(f"Loaded weights for the following layers:\n{layers}")

    def forward(self, z):
        return self.network(z)

    def loss(self, y_hat, y):
        if not self.use_blt_loss:
            return self.loss_func(y_hat, y)
        if self.use_blt_loss:
            loss = self.loss_func(y_hat[0], y.long())
            if self.timesteps > 1:
                for t in range(self.timesteps - 1):
                    loss = loss + self.loss_func(y_hat[t + 1], y.long())
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if not self.automatic_optimization:
            opt = self.optimizers()
            opt.zero_grad()

        loss = self.loss(y_hat, y)

        if not self.automatic_optimization:
            if self.scalar is None:
                self.manual_backward(loss)
                opt.step()
            if self.scalar is not None:
                self.scalar.scale(loss).backward()
                self.scalar.step(opt)
                self.scalar.update()

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

        return loss

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        print(f"{prefix}_loss: {loss}")

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


class AutoLightningModel(LightningModule):
    def __init__(
        self,
        network: CloudAutoEncoder,
        loss_func: torch.nn.Module,
        optim_func: optim,
        scheduler: schedulers = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["network", "loss_func", "optim_func", "scheduler"]
        )

        self.network = network
        self.loss_func = loss_func
        self.optim_func = optim_func
        self.scheduler = scheduler

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
                param_dict[layer].data.copy_(
                    state_dict["network." + layer].data
                )
                layers.append(layer)
            except (RuntimeError, KeyError) as e:
                print(f"Error at layer {layer}:\n{e}")

        self.network.load_state_dict(param_dict)

        if verbose:
            print(f"Loaded weights for the following layers:\n{layers}")

    def loss(self, y_hat, y):
        return self.loss_func(y_hat, y)

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
        loss = self.loss(y_hat, inputs)
        print(f"{prefix}_loss: {loss}")

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
        dataloader_inf: DataLoader,
        optim_func: optim,
        devices,
        accelerator,
        scheduler: schedulers = None,
        gamma: int = 1,
        update_interval: int = 4,
        divergence_tolerance: float = 1e-2,
        mem_percent: int = 40,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "autoencoder",
                "loss_func",
                "cluster_loss_func",
                "dataloader_inf",
                "optim_func",
                "scheduler",
            ]
        )

        self.network = network
        self.loss_func = loss_func
        self.cluster_loss_func = cluster_loss_func
        self.dataloader_inf = dataloader_inf
        self.optim_func = optim_func
        self.scheduler = scheduler
        self.gamma = gamma
        self.update_interval = update_interval
        self.divergence_tolerance = divergence_tolerance
        self.devices = devices
        self.accelerator = accelerator
        self.mem_percent = mem_percent

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
                param_dict[layer].data.copy_(
                    state_dict["network." + layer].data
                )
                layers.append(layer)
            except (RuntimeError, KeyError) as e:
                print(f"Error at layer {layer}:\n{e}")

        self.network.load_state_dict(param_dict)

        if verbose:
            print(f"Loaded weights for the following layers:\n{layers}")

    def forward(self, z):
        return self.network(z)

    def encoder_loss(self, y_hat, y):
        return self.loss_func(y_hat, y)

    def cluster_loss(self, clusters, tar_dist):
        return self.cluster_loss_func(torch.log(clusters), tar_dist)

    def training_step(self, batch, batch_idx):
        batch_size = batch.shape[0]

        if (
            self.current_epoch > 0
            and self.current_epoch % self.update_interval == 0
        ):
            print("updating target distribution")
            distribution = Distributions(
                self.network,
                self.dataloader_inf,
                self.network.num_clusters,
                devices=self.devices,
                accelerator=self.accelerator,
                mem_percent=self.mem_percent,
            )
            distribution.get_distributions_kmeans()
            self.target_distribution = distribution.target_distribution
        tar_dist = self.target_distribution[
            ((batch_idx - 1) * batch_size) : (batch_idx * batch_size),
            :,
        ]

        inputs = batch
        outputs, features, clusters = self(inputs)

        reconstruction_loss = self.loss_func(inputs, outputs)
        cluster_loss = self.cluster_loss(
            clusters, tar_dist.to(self.compute_device)
        )
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

    def on_fit_start(self) -> None:
        device = self.network.clustering_layer.weight.device
        self.compute_device = device
        print("initializing target distribution")
        distribution = Distributions(
            self.network,
            self.dataloader_inf,
            self.network.num_clusters,
            devices=self.devices,
            accelerator=self.accelerator,
            get_kmeans=True,
            mem_percent=self.mem_percent,
        )

        distribution.get_distributions_kmeans()
        self.target_distribution = distribution.target_distribution
        self.network = distribution.network
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
        optim_func: optim,
        devices,
        accelerator,
        scheduler: schedulers = None,
        gamma: int = 1,
        update_interval: int = 4,
        divergence_tolerance: float = 1e-2,
        mem_percent: int = 40,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "autoencoder",
                "loss_func",
                "cluster_loss_func",
                "optim_func",
                "scheduler",
            ]
        )

        self.network = network
        self.loss_func = loss_func
        self.cluster_loss_func = cluster_loss_func
        self.optim_func = optim_func
        self.scheduler = scheduler
        self.gamma = gamma
        self.update_interval = update_interval
        self.divergence_tolerance = divergence_tolerance
        self.devices = devices
        self.accelerator = accelerator
        self.mem_percent = mem_percent

    def forward(self, z):
        return self.network(z)

    def encoder_loss(self, y_hat, y):
        return self.loss_func(y_hat, y)

    def cluster_loss(self, clusters, tar_dist):
        return self.cluster_loss_func(torch.log(clusters), tar_dist)

    def predict_step(self, batch, batch_idx):
        return self(batch)

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
            self.default_root_dir = (
                Path(self.ckpt_path).absolute().parent.as_posix()
            )
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
        dataset: Dataset,
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
        enable_checkpointing: bool = True,
        callbacks: List[Callback] = None,
        scheduler: schedulers = None,
        logger: Logger = None,
        **kwargs,
    ):
        self.dataset = dataset

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

        self.hparams = {
            "loss_func": self.loss_func,
            "model_func": self.model_func,
            "min_epochs": self.min_epochs,
            "epochs": self.epochs,
            "optim_func": self.optim_func,
            "scheduler": self.scheduler,
            "train_val_test_split": self.train_val_test_split,
            "batch_size": self.batch_size,
            "dataset": self.dataset,
        }
        self.hparams.update(kwargs=kwargs)

    def _train_model(self):
        self.model = LightningModel(
            self.model_func, self.loss_func, self.optim_func, self.scheduler
        )

        self.datas = LightningData(hparams=self.hparams)
        self.datas.setup("fit")
        self.default_root_dir = (
            Path(self.model_save_file).absolute().parent.as_posix()
        )
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


class AutoLightningTrain:
    def __init__(
        self,
        dataset: Dataset,
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
        enable_checkpointing: bool = True,
        callbacks: List[Callback] = None,
        scheduler: schedulers = None,
        logger: Logger = None,
        **kwargs,
    ):
        self.dataset = dataset

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

        self.hparams = {
            "loss_func": self.loss_func,
            "model_func": self.model_func,
            "min_epochs": self.min_epochs,
            "epochs": self.epochs,
            "optim_func": self.optim_func,
            "scheduler": self.scheduler,
            "train_val_test_split": self.train_val_test_split,
            "batch_size": self.batch_size,
            "dataset": self.dataset,
        }
        self.hparams.update(kwargs=kwargs)

    def _train_model(self):
        self.model = AutoLightningModel(
            self.model_func, self.loss_func, self.optim_func, self.scheduler
        )

        self.datas = LightningData(hparams=self.hparams)
        self.datas.setup("fit")
        self.default_root_dir = (
            Path(self.model_save_file).absolute().parent.as_posix()
        )
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
        dataset: Dataset,
        loss_func: torch.nn.Module,
        cluster_loss_func: torch.nn.Module,
        network: DeepEmbeddedClustering,
        optim_func: optimizers._Optimizer,
        model_save_file: str,
        ckpt_file: str = None,
        train_val_test_split: List = [95, 2.5, 2.5],
        gamma: int = 1,
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
        self.dataset = dataset

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

        self.hparams = {
            "loss_func": self.loss_func,
            "network": self.network,
            "min_epochs": self.min_epochs,
            "epochs": self.epochs,
            "optim_func": self.optim_func,
            "scheduler": self.scheduler,
            "train_val_test_split": self.train_val_test_split,
            "batch_size": self.batch_size,
            "dataset": self.dataset,
        }
        self.hparams.update(kwargs=kwargs)

    def _train_model(self):
        self.datas = LightningData(hparams=self.hparams)
        self.datas.setup("fit")
        train_dataloaders = self.datas.train_dataloader()
        val_dataloaders = self.datas.val_dataloader()

        self.datas.batch_size = 1
        # train_dataloaders_inf = self.datas.train_dataloader()
        val_dataloaders_inf = self.datas.val_dataloader()

        self.model = ClusterLightningModel(
            self.network,
            self.loss_func,
            self.cluster_loss_func,
            val_dataloaders_inf,
            self.optim_func,
            self.devices,
            self.accelerator,
            self.scheduler,
            gamma=self.gamma,
            mem_percent=self.mem_percent,
        )

        self.default_root_dir = (
            Path(self.model_save_file).absolute().parent.as_posix()
        )
        self.default_root_dir = os.path.join(
            self.default_root_dir, Path(self.model_save_file).stem
        )
        Path(self.default_root_dir).mkdir(exist_ok=True)
        print("Starting training...")
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
                train_dataloaders=train_dataloaders,
                val_dataloaders=val_dataloaders,
                ckpt_path=self.ckpt_file,
            )

            self.trainer.validate(
                model=self.model,
                dataloaders=val_dataloaders,
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


class Distributions(LightningModule):
    def __init__(
        self,
        network: DeepEmbeddedClustering,
        loss_func,
        cluster_loss_func,
        optim_func,
        devices,
        accelerator,
        scheduler,
        gamma,
        dataloader,
        num_clusters,
        get_kmeans=False,
        n_init=20,
        mem_percent=20,
    ):
        super().__init__()
        self.network = network
        self.dataloader = dataloader
        self.get_kmeans = get_kmeans
        self.n_clusters = num_clusters
        self.n_init = n_init
        self.devices = devices
        self.accelerator = accelerator
        self.mem_percent = mem_percent
        self.loss_func = loss_func
        self.cluster_loss_func = cluster_loss_func
        self.optim_func = optim_func
        self.devices = devices
        self.accelerator = accelerator
        self.scheduler = scheduler
        self.gamma = gamma

    def get_distributions_kmeans(self):
        torch.cuda.empty_cache()
        cluster_distribution = None
        feature_array = None
        if self.get_kmeans:
            print("Getting KMeans")
            km = KMeans(n_clusters=self.n_clusters, n_init=self.n_init)

        local_trainer = Trainer(
            devices=self.devices, accelerator=self.accelerator
        )
        lightning_model = ClusterLightningDistModel(
            self.network,
            self.loss_func,
            self.cluster_loss_func,
            self.optim_func,
            self.devices,
            self.accelerator,
            self.scheduler,
            gamma=self.gamma,
            mem_percent=self.mem_percent,
        )
        results = local_trainer.predict(lightning_model, self.dataloader)
        outputs, features, clusters = zip(*results)

        if cluster_distribution is not None:
            cluster_distribution = torch.cat(
                (cluster_distribution, clusters), 0
            )
        else:
            cluster_distribution = clusters
        if feature_array is not None:
            feature_array = torch.cat((feature_array, features), 0)
        else:
            feature_array = features

        if self.get_kmeans:
            km.fit_predict(feature_array.detach().numpy())
            weights = torch.from_numpy(km.cluster_centers_)
            self.network.clustering_layer.set_weight(weights)
        self.predictions = torch.argmax(cluster_distribution.data, axis=1)
        self.cluster_distribution = cluster_distribution

        tar_dist = self.cluster_distribution**2 / torch.sum(
            self.cluster_distribution, axis=0
        )
        self.target_distribution = torch.transpose(
            torch.transpose(tar_dist, 0, 1) / torch.sum(tar_dist, axis=1), 0, 1
        )

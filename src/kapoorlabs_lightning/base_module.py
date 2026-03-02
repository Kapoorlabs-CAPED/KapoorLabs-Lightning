import json
import os
import torch
import torch.nn as nn
from torch import optim
from collections import OrderedDict
from lightning import LightningModule
from kapoorlabs_lightning.utils import get_most_recent_file
from . import schedulers
from .schedulers import (
    CosineAnnealingScheduler,
    ExponentialLR,
    MultiStepLR,
    WarmCosineAnnealingLR,
)

class BaseModule(LightningModule):

    def __init__(
        self,
        network: nn.Module,
        loss_func: nn.Module,
        optim_func: optim,
        scheduler: schedulers = None,
        automatic_optimization: bool = True,
        on_step: bool = True,
        on_epoch: bool = True,
        sync_dist: bool = True,
        rank_zero_only: bool = False,
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

    def forward(self, x):
        return self.network(x)

    def loss(self, y_hat, y):
        return self.loss_func(y_hat, y)

    def log_metrics(self, name: str, value, on_step: bool = None, on_epoch: bool = None):
        if on_step is None:
            on_step = self.on_step
        if on_epoch is None:
            on_epoch = self.on_epoch

        self.log(
            name,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            logger=True,
            sync_dist=self.sync_dist,
            rank_zero_only=self.rank_zero_only,
        )

    def configure_optimizers(self):
        optimizer = self.optim_func(self.parameters())

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return OrderedDict({
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "validation_loss",
                    "frequency": 1,
                },
            })
        return {"optimizer": optimizer}

    
def _restore_schedulers(scheduler, checkpoint):
    if isinstance(scheduler, WarmCosineAnnealingLR):
        t_max = checkpoint["lr_schedulers"][0]["_schedulers"][1]["T_max"]
        eta_min = checkpoint["lr_schedulers"][0]["_schedulers"][1]["eta_min"]
        t_warmup = checkpoint["lr_schedulers"][0]["_schedulers"][0][
            "total_iters"
        ]
        factor = checkpoint["lr_schedulers"][0]["_schedulers"][0][
            "start_factor"
        ]
        scheduler = WarmCosineAnnealingLR(
            t_warmup=t_warmup,
            t_max=t_max,
            eta_min=eta_min,
            factor=factor,
        )
    if isinstance(scheduler, CosineAnnealingScheduler):
        t_max = checkpoint["lr_schedulers"][0]["_schedulers"][1]["T_max"]
        eta_min = checkpoint["lr_schedulers"][0]["_schedulers"][1]["eta_min"]
        scheduler = CosineAnnealingScheduler(t_max=t_max, eta_min=eta_min)
    if isinstance(scheduler, ExponentialLR):
        gamma = checkpoint["lr_schedulers"][0]["gamma"]
        scheduler = scheduler(gamma=gamma)
    if isinstance(scheduler, MultiStepLR):
        milestones = checkpoint["lr_schedulers"][0]["milestones"]
        gamma = checkpoint["lr_schedulers"][0]["gamma"]
        scheduler = scheduler(milestones=milestones, gamma=gamma)

    return scheduler


def parse_checkpoint_json(checkpoint_model_json):
    """Parse the checkpoint JSON file or dictionary and return data."""
    assert isinstance(
        checkpoint_model_json, (str, dict)
    ), "checkpoint_model_json must be a JSON string or a dictionary"

    if isinstance(checkpoint_model_json, str):
        with open(checkpoint_model_json) as file:
            return json.load(file)
    return checkpoint_model_json


def load_checkpoint_file(model_path, preffered_checkpoint, map_location, weights_only = False):
    """Load the checkpoint file and return the checkpoint and its path."""
    if preffered_checkpoint is None:
        if os.path.isdir(model_path):
                preffered_checkpoint = get_most_recent_file(model_path, ".ckpt")
        elif os.path.isfile(model_path):
                preffered_checkpoint = model_path
        else:
                raise ValueError(f"Invalid model_path: {model_path}. It must be a valid file or directory.")
        
    checkpoint = torch.load(preffered_checkpoint, map_location=map_location, weights_only = weights_only)
    
    return checkpoint, preffered_checkpoint


def extract_learning_rate(checkpoint):
    """Extract the learning rate from a checkpoint."""
    
    try:
        return checkpoint["lr_schedulers"][0]["_last_lr"][0]
    except Exception:
        return 0.001


def restore_scheduler(scheduler, checkpoint):
    """Restore scheduler from checkpoint."""
    return _restore_schedulers(scheduler, checkpoint)


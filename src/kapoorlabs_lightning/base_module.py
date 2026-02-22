import json
import torch
import torch.nn as nn
from torch import optim
from collections import OrderedDict
from lightning import LightningModule
from kapoorlabs_lightning.utils import get_most_recent_file
from kapoorlabs_lightning import schedulers


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

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        learning_rate = self.optimizers().param_groups[0]["lr"]
        self.log_metrics("learning_rate", learning_rate)

        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics.get("validation_loss", 0))

    @classmethod
    def load_checkpoint(
        cls,
        network: nn.Module,
        checkpoint_path: str = None,
        checkpoint_dir: str = None,
        scheduler: schedulers = None,
        map_location: str = "cuda",
        **kwargs
    ):
        if checkpoint_path is None:
            if checkpoint_dir is None:
                raise ValueError("Either checkpoint_path or checkpoint_dir must be provided")
            checkpoint_path = get_most_recent_file(checkpoint_dir, ".ckpt")

        checkpoint = torch.load(checkpoint_path, map_location=map_location)



        lightning_model = cls.load_from_checkpoint(
            checkpoint_path,
            network=network,
            scheduler=scheduler,
            map_location=map_location,
            **kwargs
        )

        return lightning_model, lightning_model.network

    @classmethod
    def extract_json(cls, json_path_or_dict):
        if json_path_or_dict is None:
            return None

        if isinstance(json_path_or_dict, dict):
            return json_path_or_dict

        if isinstance(json_path_or_dict, str):
            with open(json_path_or_dict) as f:
                return json.load(f)

        raise ValueError("json_path_or_dict must be a str path or dict")

    def load_pretrained(self, pretrained_file, strict=True, verbose=True):
        if isinstance(pretrained_file, (list, tuple)):
            pretrained_file = pretrained_file[0]

        state_dict = torch.load(pretrained_file)["state_dict"]

        if not isinstance(state_dict, dict):
            state_dict = dict(state_dict)

        param_dict = dict(self.network.named_parameters())

        loaded_layers = []
        for layer in param_dict:
            if strict and "network." + layer not in state_dict:
                if verbose:
                    print(f'Could not find weights for layer "{layer}"')
                continue
            try:
                param_dict[layer].data.copy_(state_dict["network." + layer].data)
                loaded_layers.append(layer)
            except (RuntimeError, KeyError) as e:
                print(f"Error at layer {layer}: {e}")

        self.network.load_state_dict(param_dict)

        if verbose:
            print(f"Loaded weights for {len(loaded_layers)} layers")

        return loaded_layers

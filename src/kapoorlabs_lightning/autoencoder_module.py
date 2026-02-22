import torch
from torch import optim
from typing import Any
from .base_module import BaseModule
from kapoorlabs_lightning import schedulers


class AutoEncoderModule(BaseModule):

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
        scale_z: float = 1.0,
        scale_xy: float = 1.0,
    ):
        super().__init__(
            network=network,
            loss_func=loss_func,
            optim_func=optim_func,
            scheduler=scheduler,
            automatic_optimization=automatic_optimization,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only,
        )

        self.scale_z = scale_z
        self.scale_xy = scale_xy

    def training_step(self, batch, batch_idx):
        inputs = batch
        outputs, features = self(inputs)

        loss = self.loss_func(inputs, outputs)

        self.log_metrics("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "validation")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        inputs = batch
        outputs, features = self(inputs)
        loss = self.loss_func(inputs, outputs)

        self.log_metrics(f"{prefix}_loss", loss)

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

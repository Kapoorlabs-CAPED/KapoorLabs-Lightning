import torch
from torch import optim
from torchmetrics import Accuracy
from .base_module import BaseModule
from kapoorlabs_lightning import schedulers


class CellFateModule(BaseModule):

    def __init__(
        self,
        network: torch.nn.Module,
        loss_func: torch.nn.Module = None,
        optim_func: optim = None,
        scheduler: schedulers = None,
        automatic_optimization: bool = True,
        on_step: bool = True,
        on_epoch: bool = True,
        sync_dist: bool = True,
        rank_zero_only: bool = False,
        num_classes: int = 3,
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
        self.num_classes = num_classes

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)

        self.log_metrics("train_loss", loss)

        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log_metrics("learning_rate", current_lr)

        accuracy = self._compute_accuracy(y_hat, y)
        self.log_metrics("train_accuracy", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)

        self.log_metrics(f"{prefix}_loss", loss)

        accuracy = self._compute_accuracy(y_hat, y)
        self.log_metrics(f"{prefix}_accuracy", accuracy)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, *rest = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        pred_classes = torch.argmax(probs, dim=1)
        return {"logits": logits, "probabilities": probs, "predictions": pred_classes}

    def _compute_accuracy(self, outputs, labels):
        predicted_classes = torch.argmax(outputs, dim=1)
        accuracy = Accuracy(task="multiclass", num_classes=self.num_classes).to(
            self.device
        )
        return accuracy(predicted_classes, labels)

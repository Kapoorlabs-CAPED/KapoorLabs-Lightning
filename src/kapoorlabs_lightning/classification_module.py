import torch
from torch import optim
from torchmetrics import Accuracy, MeanSquaredError, MeanAbsoluteError
from .base_module import BaseModule
from kapoorlabs_lightning import schedulers


class ClassificationModule(BaseModule):

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
        num_classes: int = 2,
        oneat_accuracy: bool = False,
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
        self.oneat_accuracy = oneat_accuracy

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        self.log_metrics("train_loss", loss)

        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log_metrics("learning_rate", current_lr)

        accuracy = self.compute_accuracy(y_hat, y)
        self._log_accuracy(accuracy, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "validation")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        accuracy = self.compute_accuracy(y_hat, y)

        self.log_metrics(f"{prefix}_loss", loss)
        self._log_accuracy(accuracy, prefix)

    def _log_accuracy(self, accuracy, prefix):
        if self.oneat_accuracy:
            class_accuracy, xyz_accuracy, hwd_accuracy, confidence_accuracy = accuracy
            self.log_metrics(f"{prefix}_class_accuracy", class_accuracy)
            self.log_metrics(f"{prefix}_xyz_accuracy", xyz_accuracy)
            self.log_metrics(f"{prefix}_hwd_accuracy", hwd_accuracy)
            self.log_metrics(f"{prefix}_confidence_accuracy", confidence_accuracy)
        else:
            self.log_metrics(f"{prefix}_accuracy", accuracy)

    def compute_accuracy(self, outputs, labels):
        if self.oneat_accuracy:
            outputs = outputs.reshape(labels.shape)
            predicted_classes = outputs[:, : self.num_classes]
            true_classes = labels[:, : self.num_classes]
            predicted_xyz = outputs[:, self.num_classes : self.num_classes + 3]
            true_xyz = labels[:, self.num_classes : self.num_classes + 3]
            predicted_hwd = outputs[:, self.num_classes + 3 : self.num_classes + 6]
            true_hwd = labels[:, self.num_classes + 3 : self.num_classes + 6]
            predicted_confidence = outputs[:, self.num_classes + 6]
            true_confidence = labels[:, self.num_classes + 6]

            predicted_class_indices = torch.argmax(predicted_classes, dim=1)
            true_class_indices = torch.argmax(true_classes, dim=1)

            class_accuracy_metric = Accuracy(
                task="multiclass", num_classes=self.num_classes
            ).to(self.device)
            class_accuracy = class_accuracy_metric(
                predicted_class_indices, true_class_indices
            )

            xyz_accuracy_metric = MeanSquaredError().to(self.device)
            xyz_accuracy = 1.0 - xyz_accuracy_metric(predicted_xyz, true_xyz)

            hwd_accuracy_metric = MeanSquaredError().to(self.device)
            hwd_accuracy = 1.0 - hwd_accuracy_metric(predicted_hwd, true_hwd)

            confidence_accuracy_metric = MeanAbsoluteError().to(self.device)
            confidence_accuracy = 1.0 - confidence_accuracy_metric(
                predicted_confidence, true_confidence
            )

            return (class_accuracy, xyz_accuracy, hwd_accuracy, confidence_accuracy)
        else:
            accuracy = Accuracy(task="multiclass", num_classes=self.num_classes).to(
                self.device
            )
            return accuracy(outputs, labels)

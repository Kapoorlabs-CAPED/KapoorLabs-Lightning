import torch
import numpy as np
from torch import optim
from torchmetrics import Accuracy, MeanSquaredError, MeanAbsoluteError
from .base_module import BaseModule
from kapoorlabs_lightning import schedulers
from scipy.ndimage import center_of_mass


class OneatActionModule(BaseModule):

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
        num_classes: int = 2,
        oneat_accuracy: bool = False,
        # Prediction parameters
        imagex: int = 64,
        imagey: int = 64,
        imagez: int = 8,
        size_tminus: int = 1,
        size_tplus: int = 1,
        event_names: list = None,
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

        # Prediction parameters
        self.imagex = imagex
        self.imagey = imagey
        self.imagez = imagez
        self.size_tminus = size_tminus
        self.size_tplus = size_tplus
        self.imaget = size_tminus + size_tplus + 1
        self.event_names = event_names if event_names is not None else [f'class_{i}' for i in range(num_classes)]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # Get loss with components
        loss, loss_xyzt_hwd, loss_conf, loss_class = self.loss_func(y_hat, y)

        # Log all loss components
        self.log_metrics("train_loss", loss)
        self.log_metrics("train_loss_xyzt_hwd", loss_xyzt_hwd)
        self.log_metrics("train_loss_conf", loss_conf)
        self.log_metrics("train_loss_class", loss_class)

        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log_metrics("learning_rate", current_lr)

        accuracy = self.compute_accuracy(y_hat, y)
        self._log_accuracy(accuracy, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        x, y = batch
        y_hat = self(x)

        # Get loss with components
        loss, loss_xyzt_hwd, loss_conf, loss_class = self.loss_func(y_hat, y)

        # Log all loss components
        self.log_metrics(f"{prefix}_loss", loss)
        self.log_metrics(f"{prefix}_loss_xyzt_hwd", loss_xyzt_hwd)
        self.log_metrics(f"{prefix}_loss_conf", loss_conf)
        self.log_metrics(f"{prefix}_loss_class", loss_class)

        accuracy = self.compute_accuracy(y_hat, y)
        self._log_accuracy(accuracy, prefix)

    def _log_accuracy(self, accuracy, prefix):
        if self.oneat_accuracy:
            class_accuracy, xyzt_accuracy, hwd_accuracy, confidence_accuracy = accuracy
            self.log_metrics(f"{prefix}_class_accuracy", class_accuracy)
            self.log_metrics(f"{prefix}_xyzt_accuracy", xyzt_accuracy)
            self.log_metrics(f"{prefix}_hwd_accuracy", hwd_accuracy)
            self.log_metrics(f"{prefix}_confidence_accuracy", confidence_accuracy)
        else:
            self.log_metrics(f"{prefix}_accuracy", accuracy)

    def compute_accuracy(self, outputs, labels):
        # Squeeze spatial dimensions from outputs if present
        if outputs.dim() > 2:
            outputs = outputs.squeeze(-1).squeeze(-1).squeeze(-1)

        # Model output format: [categories (softmax), box_vector (sigmoid)]
        # GT label format: [box_vector, categories (one-hot)]
        box_vector_len = 8

        # Extract predictions: categories first, then box_vector
        predicted_classes = outputs[:, :self.num_classes]
        predicted_xyzt = outputs[:, self.num_classes:self.num_classes + 4]
        predicted_hwd = outputs[:, self.num_classes + 4:self.num_classes + 7]
        predicted_confidence = outputs[:, self.num_classes + 7]

        # Extract GT: box_vector first, then categories
        true_xyzt = labels[:, :4]
        true_hwd = labels[:, 4:7]
        true_confidence = labels[:, 7]
        true_classes = labels[:, box_vector_len:]

        predicted_class_indices = torch.argmax(predicted_classes, dim=1)
        true_class_indices = torch.argmax(true_classes, dim=1)

        if self.oneat_accuracy:
            # Full ONEAT accuracy with box metrics
            class_accuracy_metric = Accuracy(
                task="multiclass", num_classes=self.num_classes
            ).to(self.device)
            class_accuracy = class_accuracy_metric(
                predicted_class_indices, true_class_indices
            )

            xyzt_accuracy_metric = MeanSquaredError().to(self.device)
            xyzt_accuracy = 1.0 - xyzt_accuracy_metric(predicted_xyzt, true_xyzt)

            hwd_accuracy_metric = MeanSquaredError().to(self.device)
            hwd_accuracy = 1.0 - hwd_accuracy_metric(predicted_hwd, true_hwd)

            confidence_accuracy_metric = MeanAbsoluteError().to(self.device)
            confidence_accuracy = 1.0 - confidence_accuracy_metric(
                predicted_confidence, true_confidence
            )

            return (class_accuracy, xyzt_accuracy, hwd_accuracy, confidence_accuracy)
        else:
            # Simple classification accuracy
            accuracy = Accuracy(task="multiclass", num_classes=self.num_classes).to(
                self.device
            )
            return accuracy(predicted_class_indices, true_class_indices)

    def predict_step(self, batch, batch_idx):
        """
        Prediction step for ONEAT model.
        Takes raw and seg images, extracts patches for each cell, and returns predictions.

        batch format: (raw_image, seg_image, timepoint, metadata)
        raw_image: (imaget, Z, Y, X) - temporal window around current timepoint
        seg_image: (imaget, Z, Y, X) - corresponding segmentation
        timepoint: scalar - current timepoint being processed
        metadata: dict with image info
        """
        temporal_raw, temporal_seg, timepoint, metadata = batch

        # temporal_raw and temporal_seg have shape (B, T, Z, Y, X) where B=1 for prediction
        temporal_raw = temporal_raw[0]  # (T, Z, Y, X)
        temporal_seg = temporal_seg[0]  # (T, Z, Y, X)
        t = timepoint.item()

        all_detections = []

        # Find all cell instances in seg image at current timepoint (middle of temporal window)
        seg_labels = torch.unique(temporal_seg[self.size_tminus])
        seg_labels = seg_labels[seg_labels > 0]  # Remove background

        for cell_id in seg_labels:
            cell_id_val = cell_id.item()

            # Get cell mask at current timepoint
            cell_mask = (temporal_seg[self.size_tminus] == cell_id).cpu().numpy()

            # Get cell coords
            coords = np.where(cell_mask)
            if len(coords[0]) == 0:
                continue

            # Calculate center of mass
            center_coords = center_of_mass(cell_mask)
            z_center, y_center, x_center = center_coords

            # Extract patch around cell
            z_start = max(0, int(z_center - self.imagez // 2))
            z_end = min(temporal_raw.shape[1], z_start + self.imagez)
            y_start = max(0, int(y_center - self.imagey // 2))
            y_end = min(temporal_raw.shape[2], y_start + self.imagey)
            x_start = max(0, int(x_center - self.imagex // 2))
            x_end = min(temporal_raw.shape[3], x_start + self.imagex)

            # Extract patch
            patch = temporal_raw[:, z_start:z_end, y_start:y_end, x_start:x_end]

            # Pad if necessary
            if patch.shape[1:] != (self.imagez, self.imagey, self.imagex):
                padded_patch = torch.zeros((self.imaget, self.imagez, self.imagey, self.imagex),
                                          dtype=patch.dtype, device=patch.device)
                padded_patch[:, :patch.shape[1], :patch.shape[2], :patch.shape[3]] = patch
                patch = padded_patch

            # Add batch dimension
            patch_tensor = patch.unsqueeze(0)  # (1, T, Z, Y, X)

            # Predict
            with torch.no_grad():
                outputs = self(patch_tensor)  # Shape: (1, num_classes + box_vector)

            # Get predicted class (argmax, no threshold)
            predicted_class = torch.argmax(outputs[0, :self.num_classes]).item()

            # Only save non-normal events (assuming label 0 is normal)
            if predicted_class > 0:
                detection = {
                    'time': t,
                    'z': int(z_center),
                    'y': int(y_center),
                    'x': int(x_center),
                    'cell_id': cell_id_val,
                    'predicted_class': predicted_class,
                    'event_name': self.event_names[predicted_class] if predicted_class < len(self.event_names) else f'class_{predicted_class}',
                    'filename': metadata.get('filename', 'unknown')
                }
                all_detections.append(detection)

        return all_detections

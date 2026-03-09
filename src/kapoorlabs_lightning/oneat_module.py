import numpy as np
import torch
from torch import optim
from torchmetrics import Accuracy, MeanSquaredError, MeanAbsoluteError
from scipy.ndimage import center_of_mass

from .base_module import BaseModule
from kapoorlabs_lightning import schedulers


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
        event_threshold: float = 0.5,
        nms_space: int = 10,
        nms_time: int = 2,
        batch_size_predict: int = 1000,
        # Eval transforms for prediction (same as validation)
        eval_transforms=None,
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
        self.event_names = (
            event_names
            if event_names is not None
            else [f"class_{i}" for i in range(num_classes)]
        )
        self.event_threshold = event_threshold
        self.nms_space = nms_space
        self.nms_time = nms_time
        self.batch_size_predict = batch_size_predict
        self.eval_transforms = eval_transforms

        # Buffer for online NMS across timepoints
        self._recent_detections = []
        self._total_events = 0

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
        predicted_classes = outputs[:, : self.num_classes]
        predicted_xyzt = outputs[:, self.num_classes : self.num_classes + 4]
        predicted_hwd = outputs[:, self.num_classes + 4 : self.num_classes + 7]
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
        Batches all cell patches for a given timepoint, runs a single forward pass,
        applies event confidence threshold, and performs online spatial NMS within
        a temporal window of nms_time.

        batch format: (raw_image, seg_image, timepoint, metadata)
        raw_image: (1, T, Z, Y, X) - temporal window around current timepoint
        seg_image: (1, T, Z, Y, X) - corresponding segmentation
        timepoint: (1,) - current timepoint being processed
        metadata: dict with image info (collated by DataLoader)
        """
        temporal_raw, temporal_seg, timepoint, metadata = batch

        # Remove batch dim (batch_size=1 from DataLoader)
        temporal_raw = temporal_raw[0]  # (T, Z, Y, X)
        temporal_seg = temporal_seg[0]  # (T, Z, Y, X)
        t = timepoint.item()

        # Extract filename from collated metadata
        filename = metadata.get("filename", ["unknown"])
        if isinstance(filename, (list, tuple)):
            filename = filename[0]

        # Clean old detections from buffer (only keep within nms_time window)
        self._recent_detections = [
            d for d in self._recent_detections if abs(d["time"] - t) <= self.nms_time
        ]

        # Find all cell instances in seg image at current timepoint
        seg_labels = torch.unique(temporal_seg[self.size_tminus])
        seg_labels = seg_labels[seg_labels > 0]  # Remove background

        if len(seg_labels) == 0:
            return []

        # Collect all cell patches into a batch
        patches = []
        cell_info = []

        for cell_id in seg_labels:
            cell_id_val = cell_id.item()

            # Get cell mask at current timepoint
            cell_mask = (temporal_seg[self.size_tminus] == cell_id).cpu().numpy()

            coords = np.where(cell_mask)
            if len(coords[0]) == 0:
                continue

            # Calculate center of mass
            center_coords = center_of_mass(cell_mask)
            z_center, y_center, x_center = center_coords

            # Extract patch around cell center
            z_start = max(0, int(z_center - self.imagez // 2))
            z_end = min(temporal_raw.shape[1], z_start + self.imagez)
            y_start = max(0, int(y_center - self.imagey // 2))
            y_end = min(temporal_raw.shape[2], y_start + self.imagey)
            x_start = max(0, int(x_center - self.imagex // 2))
            x_end = min(temporal_raw.shape[3], x_start + self.imagex)

            patch = temporal_raw[:, z_start:z_end, y_start:y_end, x_start:x_end]

            # Pad if patch is smaller than expected
            if patch.shape[1:] != (self.imagez, self.imagey, self.imagex):
                padded = torch.zeros(
                    (self.imaget, self.imagez, self.imagey, self.imagex),
                    dtype=patch.dtype,
                    device=patch.device,
                )
                padded[:, : patch.shape[1], : patch.shape[2], : patch.shape[3]] = patch
                patch = padded

            # Apply eval transforms
            if self.eval_transforms is not None:
                patch = self.eval_transforms(patch)

            patches.append(patch)
            cell_info.append((cell_id_val, z_center, y_center, x_center))

        if len(patches) == 0:
            return []

        # Batched forward pass - chunk to avoid OOM with many cells
        all_outputs = []
        for chunk_start in range(0, len(patches), self.batch_size_predict):
            chunk = patches[chunk_start : chunk_start + self.batch_size_predict]
            batch_tensor = torch.stack(chunk, dim=0)  # (chunk_size, T, Z, Y, X)

            with torch.no_grad():
                outputs = self(batch_tensor)

            if outputs.dim() > 2:
                outputs = outputs.squeeze(-1).squeeze(-1).squeeze(-1)

            all_outputs.append(outputs)

        all_outputs = torch.cat(
            all_outputs, dim=0
        )  # (N_cells, num_classes + box_vector)

        # Extract class probabilities and box predictions
        # NOTE: softmax is already applied inside DenseVollNet.forward(),
        # do NOT apply it again here — double softmax compresses probabilities
        # toward uniform and caps confidence at ~0.73 instead of 0.99+
        class_probs = all_outputs[:, : self.num_classes]
        # Box vector layout: [x, y, z, t, h, w, d, c] (after sigmoid, values in [0, 1])
        box_predictions = all_outputs[:, self.num_classes :]

        # Collect candidate detections that pass threshold
        candidates = []
        for i, (cell_id_val, z_center, y_center, x_center) in enumerate(cell_info):
            predicted_class = torch.argmax(class_probs[i]).item()
            confidence = class_probs[i, predicted_class].item()

            # Only keep non-background events above threshold
            if predicted_class > 0 and confidence >= self.event_threshold:
                # Extract predicted bounding box dimensions (h, w, d)
                # Indices 4, 5, 6 in box_vector correspond to h, w, d
                # Scale from [0, 1] sigmoid output to pixel dimensions
                pred_h = box_predictions[i, 4].item() * self.imagey
                pred_w = box_predictions[i, 5].item() * self.imagex
                pred_d = box_predictions[i, 6].item() * self.imagez

                # Effective sphere diameter from bounding box h, w, d
                size = round((pred_h * pred_w * pred_d) ** (1.0 / 3.0), 2)

                candidates.append(
                    {
                        "time": t,
                        "z": int(z_center),
                        "y": int(y_center),
                        "x": int(x_center),
                        "score": confidence,
                        "size": size,
                        "h": round(pred_h, 2),
                        "w": round(pred_w, 2),
                        "d": round(pred_d, 2),
                        "cell_id": cell_id_val,
                        "predicted_class": predicted_class,
                        "event_name": self.event_names[predicted_class]
                        if predicted_class < len(self.event_names)
                        else f"class_{predicted_class}",
                        "filename": filename,
                    }
                )

        # Sort candidates by confidence (highest first) for greedy NMS
        candidates.sort(key=lambda d: d["score"], reverse=True)

        # Online spatial NMS: check against recent detections buffer
        surviving = []
        for det in candidates:
            suppressed = False
            # Check against existing buffer detections
            for existing in self._recent_detections:
                if det["event_name"] != existing["event_name"]:
                    continue
                if abs(det["time"] - existing["time"]) > self.nms_time:
                    continue
                spatial_dist = np.sqrt(
                    (det["x"] - existing["x"]) ** 2
                    + (det["y"] - existing["y"]) ** 2
                    + (det["z"] - existing["z"]) ** 2
                )
                if spatial_dist < self.nms_space:
                    suppressed = True
                    break

            # Also check against already-surviving detections from this timepoint
            if not suppressed:
                for s in surviving:
                    if det["event_name"] != s["event_name"]:
                        continue
                    spatial_dist = np.sqrt(
                        (det["x"] - s["x"]) ** 2
                        + (det["y"] - s["y"]) ** 2
                        + (det["z"] - s["z"]) ** 2
                    )
                    if spatial_dist < self.nms_space:
                        suppressed = True
                        break

            if not suppressed:
                surviving.append(det)
                self._recent_detections.append(det)

        self._total_events += len(surviving)

        return surviving

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
        nms_iou_threshold: float = 0.3,
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
        self.nms_iou_threshold = nms_iou_threshold
        self.batch_size_predict = batch_size_predict
        self.eval_transforms = eval_transforms

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

    @staticmethod
    def _iou3d(box_a, box_b):
        """3D IoU between two boxes given as dicts with x/y/z + h/w/d."""
        ax0, ay0, az0 = box_a["x"] - box_a["w"] / 2.0, box_a["y"] - box_a["h"] / 2.0, box_a["z"] - box_a["d"] / 2.0
        ax1, ay1, az1 = ax0 + box_a["w"], ay0 + box_a["h"], az0 + box_a["d"]
        bx0, by0, bz0 = box_b["x"] - box_b["w"] / 2.0, box_b["y"] - box_b["h"] / 2.0, box_b["z"] - box_b["d"] / 2.0
        bx1, by1, bz1 = bx0 + box_b["w"], by0 + box_b["h"], bz0 + box_b["d"]

        inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
        inter_d = max(0.0, min(az1, bz1) - max(az0, bz0))
        inter = inter_w * inter_h * inter_d
        if inter <= 0.0:
            return 0.0
        vol_a = box_a["w"] * box_a["h"] * box_a["d"]
        vol_b = box_b["w"] * box_b["h"] * box_b["d"]
        union = vol_a + vol_b - inter
        return inter / union if union > 0 else 0.0

    def predict_step(self, batch, batch_idx):
        """Per-timepoint prediction.

        For each segmented cell at the current timepoint, extracts a patch
        around its centroid, runs a batched forward pass, thresholds by
        ``event_threshold``, and applies greedy 3D-IoU NMS *per class*. The
        highest-scoring detection in any cluster of overlapping boxes wins
        — same semantics as the original caped-ai-oneat
        ``compare_function_volume``-based NMS.

        Returns a list of detection dicts for this timepoint.
        """
        # Lightning's Trainer.predict() already calls model.eval(), but be
        # defensive — accidentally training-mode dropout/BN here would silently
        # corrupt scores.
        self.eval()

        temporal_raw, temporal_seg, timepoint, metadata = batch

        # batch_size=1 from the DataLoader: drop that outer dim.
        temporal_raw = temporal_raw[0]  # (T, Z, Y, X)
        temporal_seg = temporal_seg[0]  # (T, Z, Y, X)
        t = timepoint.item()

        filename = metadata.get("filename", ["unknown"])
        if isinstance(filename, (list, tuple)):
            filename = filename[0]

        # Background labels (== 0) excluded; one centroid per remaining label.
        seg_now = temporal_seg[self.size_tminus]
        seg_labels = torch.unique(seg_now)
        seg_labels = seg_labels[seg_labels > 0]
        if len(seg_labels) == 0:
            return []

        patches = []
        cell_info = []

        for cell_id in seg_labels:
            cell_id_val = cell_id.item()
            cell_mask = (seg_now == cell_id).cpu().numpy()
            if not cell_mask.any():
                continue

            z_center, y_center, x_center = center_of_mass(cell_mask)

            z_start = max(0, int(z_center - self.imagez // 2))
            z_end = min(temporal_raw.shape[1], z_start + self.imagez)
            y_start = max(0, int(y_center - self.imagey // 2))
            y_end = min(temporal_raw.shape[2], y_start + self.imagey)
            x_start = max(0, int(x_center - self.imagex // 2))
            x_end = min(temporal_raw.shape[3], x_start + self.imagex)

            patch = temporal_raw[:, z_start:z_end, y_start:y_end, x_start:x_end]

            if patch.shape[1:] != (self.imagez, self.imagey, self.imagex):
                padded = torch.zeros(
                    (self.imaget, self.imagez, self.imagey, self.imagex),
                    dtype=patch.dtype,
                    device=patch.device,
                )
                padded[:, : patch.shape[1], : patch.shape[2], : patch.shape[3]] = patch
                patch = padded

            if self.eval_transforms is not None:
                patch = self.eval_transforms(patch)

            patches.append(patch)
            cell_info.append((cell_id_val, z_center, y_center, x_center))

        if len(patches) == 0:
            return []

        # Chunked forward pass to bound memory; one stack + one forward per chunk.
        all_outputs = []
        for chunk_start in range(0, len(patches), self.batch_size_predict):
            chunk = patches[chunk_start : chunk_start + self.batch_size_predict]
            batch_tensor = torch.stack(chunk, dim=0)
            with torch.no_grad():
                outputs = self(batch_tensor)
            if outputs.dim() > 2:
                outputs = outputs.squeeze(-1).squeeze(-1).squeeze(-1)
            all_outputs.append(outputs)
        all_outputs = torch.cat(all_outputs, dim=0)

        # Softmax already applied in DenseVollNet.forward(); don't double-apply.
        class_probs = all_outputs[:, : self.num_classes]
        box_predictions = all_outputs[:, self.num_classes :]

        # Box layout: [x, y, z, t, h, w, d, c]; sigmoid'd inside the model.
        candidates = []
        for i, (cell_id_val, z_center, y_center, x_center) in enumerate(cell_info):
            predicted_class = torch.argmax(class_probs[i]).item()
            score = class_probs[i, predicted_class].item()

            if predicted_class <= 0 or score < self.event_threshold:
                continue

            pred_h = box_predictions[i, 4].item() * self.imagey
            pred_w = box_predictions[i, 5].item() * self.imagex
            pred_d = box_predictions[i, 6].item() * self.imagez
            size = round((pred_h * pred_w * pred_d) ** (1.0 / 3.0), 2)

            candidates.append(
                {
                    "time": t,
                    "z": int(z_center),
                    "y": int(y_center),
                    "x": int(x_center),
                    "score": score,
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

        if not candidates:
            return []

        # Per-class greedy IoU NMS: sort by score desc, take top, suppress
        # remaining boxes whose 3D IoU with the kept box >= nms_iou_threshold.
        # Mirrors the caped-ai-oneat compare_function_volume semantics.
        by_class = {}
        for det in candidates:
            by_class.setdefault(det["event_name"], []).append(det)

        surviving = []
        for event_name, group in by_class.items():
            group.sort(key=lambda d: d["score"], reverse=True)
            kept = []
            for det in group:
                if any(self._iou3d(det, k) >= self.nms_iou_threshold for k in kept):
                    continue
                kept.append(det)
            surviving.extend(kept)

        self._total_events += len(surviving)
        return surviving

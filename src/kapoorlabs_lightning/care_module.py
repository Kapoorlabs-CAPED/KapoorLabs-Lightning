"""
CARE (Content-Aware image REstoration) Lightning module.

Supervised denoising: predicts high SNR from low SNR 3D volumes.
Uses UNet from careamics package.
"""

import numpy as np
import torch
from torch import optim

from .base_module import BaseModule
from kapoorlabs_lightning import schedulers


class CareModule(BaseModule):
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
        # Prediction parameters
        n_tiles: list = None,
        tile_overlap: float = 0.125,
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

        self.n_tiles = n_tiles if n_tiles is not None else [1, 4, 4]
        self.tile_overlap = tile_overlap
        self.eval_transforms = eval_transforms

    def training_step(self, batch, batch_idx):
        low, high = batch

        # UNet expects (B, C, Z, Y, X) — add channel dim
        low = low.unsqueeze(1)
        high = high.unsqueeze(1)

        predicted = self(low)
        loss = self.loss_func(predicted, high)

        self.log_metrics("train_loss", loss)
        psnr = self._compute_psnr(predicted, high)
        self.log_metrics("train_psnr", psnr)

        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log_metrics("learning_rate", current_lr)

        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        low, high = batch
        low = low.unsqueeze(1)
        high = high.unsqueeze(1)

        predicted = self(low)
        loss = self.loss_func(predicted, high)

        self.log_metrics(f"{prefix}_loss", loss)
        psnr = self._compute_psnr(predicted, high)
        self.log_metrics(f"{prefix}_psnr", psnr)

    def predict_step(self, batch, batch_idx):
        """
        Prediction on tiled input.

        batch: (tile_tensor, coords) from CarePredictionDataset
            tile_tensor: (B, Z, Y, X)
            coords: (B, 6) — [z_start, y_start, x_start, tile_z, tile_y, tile_x]
        """
        tiles, coords = batch

        # Add channel dim: (B, 1, Z, Y, X)
        tiles = tiles.unsqueeze(1)

        with torch.no_grad():
            predicted = self(tiles)

        # Remove channel dim: (B, Z, Y, X)
        predicted = predicted.squeeze(1)

        return predicted.cpu(), coords.cpu()

    def _compute_psnr(self, predicted, target, max_val=1.0):
        """Compute Peak Signal-to-Noise Ratio."""
        mse = torch.mean((predicted - target) ** 2)
        if mse == 0:
            return torch.tensor(float("inf"))
        return 10 * torch.log10(max_val**2 / mse)


def stitch_tiles(predictions, volume_shape, overlap_fraction=0.125):
    """
    Stitch predicted tiles back into a full volume using linear blending.

    Works for any spatial dimensionality (2D YX or 3D ZYX).

    Args:
        predictions: Iterable of ``(predicted_tile, coords)`` pairs from
            ``predict_step``. ``predicted_tile`` is a ``(B, *spatial)``
            tensor — or ``(B, C, *spatial)`` with a channel axis;
            channel 0 is taken in that case. ``coords`` is a
            ``(B, 2 * ndim)`` long tensor laid out as
            ``[start_0, …, start_n, size_0, …, size_n]``.
        volume_shape: shape of the full output volume — any length.
        overlap_fraction: Overlap as fraction of tile size.

    Returns:
        numpy array of shape ``volume_shape`` — the stitched volume.
    """
    output = np.zeros(volume_shape, dtype=np.float32)
    weight = np.zeros(volume_shape, dtype=np.float32)
    ndim = len(volume_shape)

    for pred_batch, coord_batch in predictions:
        for i in range(pred_batch.shape[0]):
            tile = (
                pred_batch[i].cpu().numpy()
                if hasattr(pred_batch[i], "cpu")
                else np.asarray(pred_batch[i])
            )
            # If the model emitted a leading channel axis, take channel 0.
            if tile.ndim == ndim + 1:
                tile = tile[0]
            coords = [int(v) for v in coord_batch[i].tolist()]
            starts = coords[:ndim]
            sizes = coords[ndim:]

            sl = tuple(slice(s, s + sz) for s, sz in zip(starts, sizes))
            w = _make_blend_weight(tuple(sizes), overlap_fraction)
            output[sl] += tile * w
            weight[sl] += w

    # Normalize by total weight
    mask = weight > 0
    output[mask] /= weight[mask]

    return output


def _make_blend_weight(tile_shape, overlap_fraction):
    """Create a weight array that ramps from 0 at edges to 1 at center."""
    weight = np.ones(tile_shape, dtype=np.float32)
    for axis in range(len(tile_shape)):
        size = tile_shape[axis]
        overlap_px = max(1, int(size * overlap_fraction))
        ramp = np.linspace(0, 1, overlap_px, dtype=np.float32)

        # Build 1D weight for this axis
        w1d = np.ones(size, dtype=np.float32)
        w1d[:overlap_px] = ramp
        w1d[-overlap_px:] = ramp[::-1]

        # Broadcast to full shape
        shape = [1] * len(tile_shape)
        shape[axis] = size
        weight *= w1d.reshape(shape)

    return weight


__all__ = ["CareModule", "stitch_tiles"]

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

    Args:
        predictions: List of (predicted_tile, coords) from predict_step.
            predicted_tile: (B, Z, Y, X) tensor
            coords: (B, 6) tensor [z_start, y_start, x_start, tz, ty, tx]
        volume_shape: (Z, Y, X) shape of the full output volume.
        overlap_fraction: Overlap as fraction of tile size.

    Returns:
        numpy array (Z, Y, X) — the stitched denoised volume.
    """
    output = np.zeros(volume_shape, dtype=np.float32)
    weight = np.zeros(volume_shape, dtype=np.float32)

    for pred_batch, coord_batch in predictions:
        for i in range(pred_batch.shape[0]):
            tile = pred_batch[i].numpy()
            zs, ys, xs, tz, ty, tx = coord_batch[i].tolist()
            zs, ys, xs, tz, ty, tx = int(zs), int(ys), int(xs), int(tz), int(ty), int(tx)

            # Create blending weight (linear ramp at borders)
            w = _make_blend_weight((tz, ty, tx), overlap_fraction)

            output[zs : zs + tz, ys : ys + ty, xs : xs + tx] += tile * w
            weight[zs : zs + tz, ys : ys + ty, xs : xs + tx] += w

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

"""
Dataset classes for CARE (Content-Aware image REstoration).

H5CareDataset: Reads paired low/high SNR patches from H5 file.
CarePredictionDataset: Tiles a full 3D volume for prediction.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class H5CareDataset(Dataset):
    """
    HDF5 dataset for paired low/high SNR 3D patches.

    H5 structure:
        /{split}/low   - shape (N, Z, Y, X)
        /{split}/high  - shape (N, Z, Y, X)

    Args:
        h5_file: Path to HDF5 file.
        split: 'train' or 'val'.
        transforms: Callable that takes (low, high) tensors and returns (low, high).
    """

    def __init__(self, h5_file, split="train", transforms=None,
                 input_key="low", target_key="high"):
        self.h5_file = h5_file
        self.split = split
        self.transforms = transforms

        self.h5_handle = h5py.File(h5_file, "r", swmr=True)
        self.low_dataset = self.h5_handle[f"{split}/{input_key}"]
        self.high_dataset = self.h5_handle[f"{split}/{target_key}"]

        assert len(self.low_dataset) == len(self.high_dataset), (
            f"Mismatch: {len(self.low_dataset)} {input_key} vs {len(self.high_dataset)} {target_key} patches"
        )
        self.num_samples = len(self.low_dataset)
        self.patch_shape = self.low_dataset.shape[1:]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        low = torch.from_numpy(self.low_dataset[idx].astype(np.float32))
        high = torch.from_numpy(self.high_dataset[idx].astype(np.float32))

        if self.transforms is not None:
            low, high = self.transforms(low, high)

        return low, high

    def __del__(self):
        if hasattr(self, "h5_handle") and self.h5_handle:
            self.h5_handle.close()


class CarePredictionDataset(Dataset):
    """
    Dataset that tiles a 3D volume (ZYX) for batched prediction.

    Args:
        volume: numpy array (Z, Y, X).
        tile_shape: (tile_z, tile_y, tile_x) size of each tile.
        overlap: Overlap as fraction of tile size (e.g., 0.125).
        normalizer: Callable that normalizes a single tensor.
    """

    def __init__(self, volume, tile_shape, overlap=0.125, normalizer=None):
        self.volume = volume
        self.tile_shape = tile_shape
        self.overlap = overlap
        self.normalizer = normalizer

        self.tiles = self._compute_tile_grid()

    def _compute_tile_grid(self):
        """Compute (z_start, y_start, x_start) for each tile."""
        tiles = []
        vol_shape = self.volume.shape
        for axis_idx, (vol_size, tile_size) in enumerate(
            zip(vol_shape, self.tile_shape)
        ):
            overlap_px = int(tile_size * self.overlap)
            stride = tile_size - overlap_px
            starts = []
            pos = 0
            while pos + tile_size <= vol_size:
                starts.append(pos)
                pos += stride
            # Make sure last tile covers the end
            if not starts or starts[-1] + tile_size < vol_size:
                starts.append(max(0, vol_size - tile_size))
            if axis_idx == 0:
                tiles = [(s,) for s in starts]
            else:
                tiles = [t + (s,) for t in tiles for s in starts]
        return tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        zs, ys, xs = self.tiles[idx]
        tz, ty, tx = self.tile_shape

        tile = self.volume[zs : zs + tz, ys : ys + ty, xs : xs + tx]
        tile = torch.from_numpy(tile.astype(np.float32))

        if self.normalizer is not None:
            tile = self.normalizer(tile)

        # Return tile and its position for stitching
        coords = torch.tensor([zs, ys, xs, tz, ty, tx], dtype=torch.long)
        return tile, coords

    def get_volume_shape(self):
        return self.volume.shape

    def get_tile_shape(self):
        return self.tile_shape


def compute_tile_shape(volume_shape, n_tiles):
    """
    Compute tile size from volume shape and number of tiles per axis.

    Args:
        volume_shape: (Z, Y, X) tuple.
        n_tiles: (nz, ny, nx) number of tiles per axis.

    Returns:
        (tile_z, tile_y, tile_x) tuple.
    """
    return tuple(
        int(np.ceil(v / n)) for v, n in zip(volume_shape, n_tiles)
    )


__all__ = [
    "H5CareDataset",
    "CarePredictionDataset",
    "compute_tile_shape",
]

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
    Dataset that tiles a 2D or 3D volume for batched prediction.

    The ROI Mask-UNet (``conv_dims=2``) needs 2D tiling; CARE / nuclei /
    membrane U-Net / StarDist (``conv_dims=3``) need 3D tiling. Both
    use the same class.

    Args:
        volume: numpy array (Y, X) or (Z, Y, X).
        tile_shape: tile size, length must match volume.ndim.
        overlap: Overlap as fraction of tile size (e.g., 0.125).
        normalizer: Callable that normalizes a single tensor.

    The ``coords`` tensor yielded per item is laid out as
    ``[start_axis_0, …, start_axis_n, size_axis_0, …, size_axis_n]`` —
    the same schema :func:`kapoorlabs_lightning.care_module.stitch_tiles`
    consumes for both 2D and 3D inputs.
    """

    def __init__(self, volume, tile_shape, overlap=0.125, normalizer=None):
        if volume.ndim not in (2, 3):
            raise ValueError(
                f"CarePredictionDataset expects a 2D or 3D volume, "
                f"got ndim={volume.ndim}"
            )
        if len(tile_shape) != volume.ndim:
            raise ValueError(
                f"tile_shape {tile_shape} has {len(tile_shape)} entries "
                f"but volume.ndim={volume.ndim}"
            )
        self.volume = volume
        self.tile_shape = tuple(tile_shape)
        self.overlap = float(overlap)
        self.normalizer = normalizer

        self.tiles = self._compute_tile_grid()

    def _compute_tile_grid(self):
        """Compute per-tile start tuples for any spatial dimensionality."""
        tiles = []
        for axis_idx, (vol_size, tile_size) in enumerate(
            zip(self.volume.shape, self.tile_shape)
        ):
            overlap_px = int(tile_size * self.overlap)
            stride = max(1, tile_size - overlap_px)
            starts = []
            pos = 0
            while pos + tile_size <= vol_size:
                starts.append(pos)
                pos += stride
            # Make sure last tile covers the end.
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
        starts = self.tiles[idx]
        sl = tuple(slice(s, s + sz) for s, sz in zip(starts, self.tile_shape))
        tile = self.volume[sl]
        tile = torch.from_numpy(np.ascontiguousarray(tile, dtype=np.float32))

        if self.normalizer is not None:
            tile = self.normalizer(tile)

        # [start_0, ..., start_n, size_0, ..., size_n]
        coords = torch.tensor(
            list(starts) + list(self.tile_shape), dtype=torch.long,
        )
        return tile, coords

    def get_volume_shape(self):
        return self.volume.shape

    def get_tile_shape(self):
        return self.tile_shape


def compute_tile_shape(volume_shape, n_tiles):
    """
    Compute tile size from volume shape and number of tiles per axis.

    Args:
        volume_shape: tuple — any spatial dimensionality (2D YX or 3D ZYX).
        n_tiles: tuple of same length as ``volume_shape``.

    Returns:
        Per-axis tile size as a tuple.
    """
    if len(volume_shape) != len(n_tiles):
        raise ValueError(
            f"volume_shape {volume_shape} and n_tiles {n_tiles} must have "
            f"the same length"
        )
    return tuple(
        int(np.ceil(v / n)) for v, n in zip(volume_shape, n_tiles)
    )


__all__ = [
    "H5CareDataset",
    "CarePredictionDataset",
    "compute_tile_shape",
]

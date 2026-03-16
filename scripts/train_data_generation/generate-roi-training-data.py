"""
Generate ROI segmentation training data (H5 file) from paired 2D timelapse images.

Expects:
    raw_dir/  — raw 2D timelapse .tif files (TYX or TCYX)
    mask_dir/ — segmentation mask .tif files (TYX, same filenames)

Process:
    Each timepoint is treated as an independent 2D frame.
    Patches are extracted per-frame and written to H5 incrementally
    in batches — no large arrays held in memory.

Produces:
    H5 file with /train/raw, /train/mask, /val/raw, /val/mask
    Each dataset has shape (N_patches, patch_y, patch_x)
"""

import os
from glob import glob
from pathlib import Path

import h5py
import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from tifffile import imread

from scenario_generate_roi import RoiDataClass

configstore = ConfigStore.instance()
configstore.store(name="RoiDataClass", node=RoiDataClass)


def _write_roi_batch_to_h5(group, key, patches, dtype=None):
    """Append a batch of patches to a resizable H5 dataset."""
    arr = np.array(patches, dtype=dtype)
    if key not in group:
        group.create_dataset(
            key,
            data=arr,
            maxshape=(None,) + arr.shape[1:],
            chunks=True,
            compression="gzip",
        )
    else:
        old_size = group[key].shape[0]
        new_size = old_size + len(patches)
        group[key].resize(new_size, axis=0)
        group[key][old_size:new_size] = arr


def extract_and_write_patches(
    raw_timelapse, mask_timelapse, patch_shape, stride, group, batch_write_size=100
):
    """
    Iterate over each timepoint in a 2D timelapse, extract 2D patches,
    and write them directly to H5 in batches.

    Args:
        raw_timelapse:  (T, Y, X) numpy array, float
        mask_timelapse: (T, Y, X) numpy array, integer labels
        patch_shape:    (py, px)
        stride:         (sy, sx)
        group:          open H5 group to write into
        batch_write_size: flush to disk every N patches

    Returns:
        total number of patches written
    """
    py, px = patch_shape
    sy, sx = stride
    n_frames, vy, vx = raw_timelapse.shape

    raw_batch = []
    mask_batch = []
    total = 0

    for t in range(n_frames):
        raw_frame = raw_timelapse[t]
        mask_frame = mask_timelapse[t]

        for y in range(0, vy - py + 1, sy):
            for x in range(0, vx - px + 1, sx):
                raw_batch.append(raw_frame[y : y + py, x : x + px])
                mask_batch.append(mask_frame[y : y + py, x : x + px])

                if len(raw_batch) >= batch_write_size:
                    _write_roi_batch_to_h5(group, "raw", raw_batch,
                                           dtype=raw_timelapse.dtype)
                    _write_roi_batch_to_h5(group, "mask", mask_batch,
                                           dtype=mask_timelapse.dtype)
                    total += len(raw_batch)
                    raw_batch = []
                    mask_batch = []

    if raw_batch:
        _write_roi_batch_to_h5(group, "raw", raw_batch, dtype=raw_timelapse.dtype)
        _write_roi_batch_to_h5(group, "mask", mask_batch, dtype=mask_timelapse.dtype)
        total += len(raw_batch)

    return total


def _percentile_normalize(vol, pmin, pmax):
    """Normalize volume to [0, 1] using percentile clipping."""
    lo = np.percentile(vol, pmin)
    hi = np.percentile(vol, pmax)
    if hi - lo < 1e-8:
        return np.zeros_like(vol, dtype=np.float32)
    return ((vol.astype(np.float32) - lo) / (hi - lo)).clip(0, 1)


def _load_timelapse(filepath, pmin=None, pmax=None):
    """Load a 2D timelapse (T, Y, X), optionally percentile-normalize."""
    vol = imread(filepath)
    assert vol.ndim == 3, f"Expected TYX (3D), got shape {vol.shape} in {filepath}"
    if pmin is not None and pmax is not None:
        vol = _percentile_normalize(vol, pmin, pmax)
    return vol


@hydra.main(config_path="../conf", config_name="scenario_generate_roi", version_base="1.3")
def main(config: RoiDataClass):
    patch_y = config.parameters.patch_y
    patch_x = config.parameters.patch_x
    file_type = config.parameters.file_type
    pmin = config.parameters.pmin
    pmax = config.parameters.pmax

    base_data_dir = config.train_data_paths.base_data_dir
    raw_dir = os.path.join(base_data_dir, config.train_data_paths.raw_dir)
    mask_dir = os.path.join(base_data_dir, config.train_data_paths.mask_dir)
    h5_output_path = os.path.join(
        base_data_dir, config.train_data_paths.roi_h5_file
    )

    patch_shape = (patch_y, patch_x)
    # 2/3-patch stride (33% overlap) for training, non-overlapping for val
    train_stride = (max(1, 2 * patch_y // 3), max(1, 2 * patch_x // 3))

    # Find paired files
    raw_files = sorted(glob(os.path.join(raw_dir, file_type)))
    print(f"Found {len(raw_files)} raw timelapse files in {raw_dir}")

    if not raw_files:
        print("Error: No files found")
        return

    paired_files = []
    for raw_file in raw_files:
        basename = os.path.basename(raw_file)
        mask_file = os.path.join(mask_dir, basename)
        if os.path.exists(mask_file):
            paired_files.append((raw_file, mask_file))
        else:
            print(f"Warning: No matching mask for {basename}, skipping")

    print(f"Found {len(paired_files)} valid pairs")

    if not paired_files:
        print("Error: No valid pairs found")
        return

    # Last timelapse reserved for validation
    if len(paired_files) > 1:
        train_pairs = paired_files[:-1]
        val_pairs = paired_files[-1:]
    else:
        train_pairs = paired_files
        val_pairs = paired_files

    print(f"\nPatch shape: {patch_shape}, train stride: {train_stride}")

    Path(os.path.dirname(h5_output_path)).mkdir(parents=True, exist_ok=True)

    train_total = 0
    val_total = 0

    with h5py.File(h5_output_path, "w") as h5f:
        train_grp = h5f.create_group("train")
        val_grp = h5f.create_group("val")

        for raw_file, mask_file in train_pairs:
            basename = os.path.basename(raw_file)
            print(f"\n[TRAIN] Processing {basename}...")

            raw_tl = _load_timelapse(raw_file, pmin, pmax)
            mask_tl = _load_timelapse(mask_file)

            assert raw_tl.shape == mask_tl.shape, (
                f"Shape mismatch: raw {raw_tl.shape} vs mask {mask_tl.shape}"
            )
            print(f"  Timelapse shape: {raw_tl.shape}  (T, Y, X)")

            n = extract_and_write_patches(
                raw_tl, mask_tl, patch_shape, train_stride, train_grp
            )
            train_total += n
            print(f"  Wrote {n} patches (total: {train_total})")

        for raw_file, mask_file in val_pairs:
            basename = os.path.basename(raw_file)
            print(f"\n[VAL] Processing {basename}...")

            raw_tl = _load_timelapse(raw_file, pmin, pmax)
            mask_tl = _load_timelapse(mask_file)

            assert raw_tl.shape == mask_tl.shape

            # Non-overlapping stride for validation
            n = extract_and_write_patches(
                raw_tl, mask_tl, patch_shape, patch_shape, val_grp
            )
            val_total += n
            print(f"  Wrote {n} val patches (total: {val_total})")

    file_size_mb = os.path.getsize(h5_output_path) / (1024 * 1024)
    print(f"\nTrain: {train_total} patches, Val: {val_total} patches")
    print(f"Saved H5 file: {h5_output_path} ({file_size_mb:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()

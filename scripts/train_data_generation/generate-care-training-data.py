"""
Generate CARE training data (H5 file) from paired low/high SNR 3D images.

Expects:
    low_dir/  — low SNR .tif files (ZYX)
    high_dir/ — high SNR .tif files (ZYX, same filenames)

Produces:
    H5 file with /train/low, /train/high, /val/low, /val/high
    Each dataset has shape (N_patches, patch_z, patch_y, patch_x)

Patches are written incrementally in batches to avoid holding all data in memory.
"""

import os
from glob import glob
from pathlib import Path

import h5py
import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from tifffile import imread

from scenario_generate_care import CareDataClass

configstore = ConfigStore.instance()
configstore.store(name="CareDataClass", node=CareDataClass)


def _write_care_batch_to_h5(group, key, patches):
    """Append a batch of patches to a resizable H5 dataset."""
    arr = np.array(patches)
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
    low_vol, high_vol, patch_shape, stride, group, batch_write_size=100
):
    """
    Extract 3D patches from paired volumes and write them directly to H5
    in batches, never holding more than batch_write_size patches in memory.
    """
    pz, py, px = patch_shape
    sz, sy, sx = stride
    vz, vy, vx = low_vol.shape

    low_batch = []
    high_batch = []
    total = 0

    for z in range(0, vz - pz + 1, sz):
        for y in range(0, vy - py + 1, sy):
            for x in range(0, vx - px + 1, sx):
                low_batch.append(low_vol[z : z + pz, y : y + py, x : x + px])
                high_batch.append(high_vol[z : z + pz, y : y + py, x : x + px])

                if len(low_batch) >= batch_write_size:
                    _write_care_batch_to_h5(group, "low", low_batch)
                    _write_care_batch_to_h5(group, "high", high_batch)
                    total += len(low_batch)
                    low_batch = []
                    high_batch = []

    if len(low_batch) > 0:
        _write_care_batch_to_h5(group, "low", low_batch)
        _write_care_batch_to_h5(group, "high", high_batch)
        total += len(low_batch)

    return total


def _percentile_normalize(vol, pmin, pmax):
    """Normalize volume to [0, 1] using percentile clipping."""
    lo = np.percentile(vol, pmin)
    hi = np.percentile(vol, pmax)
    if hi - lo < 1e-8:
        return np.zeros_like(vol, dtype=np.float32)
    return ((vol.astype(np.float32) - lo) / (hi - lo)).clip(0, 1)


def _load_volume(filepath, pmin=0.1, pmax=99.9):
    """Load a 3D volume, take first channel if 4D, percentile-normalize."""
    vol = imread(filepath)
    if vol.ndim == 4:
        vol = vol[:, 0] if vol.shape[1] < vol.shape[0] else vol[0]
    vol = _percentile_normalize(vol, pmin, pmax)
    return vol


@hydra.main(config_path="../conf", config_name="scenario_generate_care")
def main(config: CareDataClass):
    patch_z = config.parameters.patch_z
    patch_y = config.parameters.patch_y
    patch_x = config.parameters.patch_x
    file_type = config.parameters.file_type
    pmin = config.parameters.pmin
    pmax = config.parameters.pmax

    base_data_dir = config.train_data_paths.base_data_dir
    low_dir = os.path.join(base_data_dir, config.train_data_paths.low_dir)
    high_dir = os.path.join(base_data_dir, config.train_data_paths.high_dir)
    h5_output_path = os.path.join(
        base_data_dir, config.train_data_paths.care_h5_file
    )

    patch_shape = (patch_z, patch_y, patch_x)
    # Use half-patch stride for 50% overlap to get more training data
    stride = (max(1, 2 * patch_z // 3), max(1, 2 * patch_y // 3), max(1, 2 * patch_x // 3))

    # Find paired files
    low_files = sorted(glob(os.path.join(low_dir, file_type)))
    print(f"Found {len(low_files)} low SNR files in {low_dir}")

    if len(low_files) == 0:
        print("Error: No files found")
        return

    # Verify pairs exist
    paired_files = []
    for low_file in low_files:
        basename = os.path.basename(low_file)
        high_file = os.path.join(high_dir, basename)
        if os.path.exists(high_file):
            paired_files.append((low_file, high_file))
        else:
            print(f"Warning: No matching high SNR file for {basename}, skipping")

    print(f"Found {len(paired_files)} valid pairs")

    if len(paired_files) == 0:
        print("Error: No valid pairs found")
        return

    # Split files into train/val (last file for validation)
    if len(paired_files) > 1:
        train_pairs = paired_files[:-1]
        val_pairs = paired_files[-1:]
    else:
        train_pairs = paired_files
        val_pairs = paired_files

    print(f"\nPatch shape: {patch_shape}, stride: {stride}")

    Path(os.path.dirname(h5_output_path)).mkdir(parents=True, exist_ok=True)

    train_total = 0
    val_total = 0

    with h5py.File(h5_output_path, "w") as h5f:
        train_grp = h5f.create_group("train")
        val_grp = h5f.create_group("val")

        for low_file, high_file in train_pairs:
            basename = os.path.basename(low_file)
            print(f"\n[TRAIN] Processing {basename}...")

            low_vol = _load_volume(low_file, pmin, pmax)
            high_vol = _load_volume(high_file, pmin, pmax)

            assert low_vol.shape == high_vol.shape, (
                f"Shape mismatch: low {low_vol.shape} vs high {high_vol.shape}"
            )
            print(f"  Volume shape: {low_vol.shape}")

            n = extract_and_write_patches(
                low_vol, high_vol, patch_shape, stride, train_grp
            )
            train_total += n
            print(f"  Wrote {n} patches (total: {train_total})")

        for low_file, high_file in val_pairs:
            basename = os.path.basename(low_file)
            print(f"\n[VAL] Processing {basename}...")

            low_vol = _load_volume(low_file, pmin, pmax)
            high_vol = _load_volume(high_file, pmin, pmax)

            assert low_vol.shape == high_vol.shape

            # Use non-overlapping stride for validation
            n = extract_and_write_patches(
                low_vol, high_vol, patch_shape, patch_shape, val_grp
            )
            val_total += n
            print(f"  Wrote {n} val patches (total: {val_total})")

    file_size_mb = os.path.getsize(h5_output_path) / (1024 * 1024)
    print(f"\nTrain: {train_total} patches, Val: {val_total} patches")
    print(f"Saved H5 file: {h5_output_path} ({file_size_mb:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()

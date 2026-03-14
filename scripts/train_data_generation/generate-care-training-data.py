"""
Generate CARE training data (H5 file) from paired low/high SNR 3D images.

Expects:
    low_dir/  — low SNR .tif files (ZYX)
    high_dir/ — high SNR .tif files (ZYX, same filenames)

Produces:
    H5 file with /train/low, /train/high, /val/low, /val/high
    Each dataset has shape (N_patches, patch_z, patch_y, patch_x)
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


def extract_patches_3d(volume, patch_shape, stride=None):
    """
    Extract non-overlapping (or strided) 3D patches from a volume.

    Args:
        volume: numpy array (Z, Y, X)
        patch_shape: (pz, py, px)
        stride: (sz, sy, sx) — defaults to patch_shape (non-overlapping)

    Returns:
        list of patches as numpy arrays
    """
    if stride is None:
        stride = patch_shape

    pz, py, px = patch_shape
    sz, sy, sx = stride
    vz, vy, vx = volume.shape

    patches = []
    for z in range(0, vz - pz + 1, sz):
        for y in range(0, vy - py + 1, sy):
            for x in range(0, vx - px + 1, sx):
                patch = volume[z : z + pz, y : y + py, x : x + px]
                patches.append(patch)

    return patches


@hydra.main(config_path="../conf", config_name="scenario_generate_care")
def main(config: CareDataClass):
    patch_z = config.parameters.patch_z
    patch_y = config.parameters.patch_y
    patch_x = config.parameters.patch_x
    file_type = config.parameters.file_type

    base_data_dir = config.train_data_paths.base_data_dir
    low_dir = os.path.join(base_data_dir, config.train_data_paths.low_dir)
    high_dir = os.path.join(base_data_dir, config.train_data_paths.high_dir)
    h5_output_path = os.path.join(
        base_data_dir, config.train_data_paths.care_h5_file
    )

    patch_shape = (patch_z, patch_y, patch_x)
    # Use half-patch stride for 50% overlap to get more training data
    stride = (max(1, patch_z // 2), max(1, patch_y // 2), max(1, patch_x // 2))

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
        # Single file: use it for both train and val
        train_pairs = paired_files
        val_pairs = paired_files

    # Extract patches
    print(f"\nPatch shape: {patch_shape}, stride: {stride}")

    train_low_patches = []
    train_high_patches = []
    val_low_patches = []
    val_high_patches = []

    for low_file, high_file in train_pairs:
        basename = os.path.basename(low_file)
        print(f"\n[TRAIN] Processing {basename}...")

        low_vol = imread(low_file)
        high_vol = imread(high_file)

        # Handle multi-channel: take first channel if needed
        if low_vol.ndim == 4:
            low_vol = low_vol[:, 0] if low_vol.shape[1] < low_vol.shape[0] else low_vol[0]
        if high_vol.ndim == 4:
            high_vol = high_vol[:, 0] if high_vol.shape[1] < high_vol.shape[0] else high_vol[0]

        assert low_vol.shape == high_vol.shape, (
            f"Shape mismatch: low {low_vol.shape} vs high {high_vol.shape}"
        )

        print(f"  Volume shape: {low_vol.shape}")

        low_patches = extract_patches_3d(low_vol, patch_shape, stride)
        high_patches = extract_patches_3d(high_vol, patch_shape, stride)

        print(f"  Extracted {len(low_patches)} patches")
        train_low_patches.extend(low_patches)
        train_high_patches.extend(high_patches)

    for low_file, high_file in val_pairs:
        basename = os.path.basename(low_file)
        print(f"\n[VAL] Processing {basename}...")

        low_vol = imread(low_file)
        high_vol = imread(high_file)

        if low_vol.ndim == 4:
            low_vol = low_vol[:, 0] if low_vol.shape[1] < low_vol.shape[0] else low_vol[0]
        if high_vol.ndim == 4:
            high_vol = high_vol[:, 0] if high_vol.shape[1] < high_vol.shape[0] else high_vol[0]

        assert low_vol.shape == high_vol.shape

        # Use non-overlapping stride for validation
        low_patches = extract_patches_3d(low_vol, patch_shape)
        high_patches = extract_patches_3d(high_vol, patch_shape)

        print(f"  Extracted {len(low_patches)} val patches")
        val_low_patches.extend(low_patches)
        val_high_patches.extend(high_patches)

    # Stack into arrays
    train_low = np.stack(train_low_patches, axis=0)
    train_high = np.stack(train_high_patches, axis=0)
    val_low = np.stack(val_low_patches, axis=0)
    val_high = np.stack(val_high_patches, axis=0)

    print(f"\nTrain: {train_low.shape[0]} patches, Val: {val_low.shape[0]} patches")
    print(f"Patch shape: {train_low.shape[1:]}")

    # Write H5 file
    Path(os.path.dirname(h5_output_path)).mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_output_path, "w") as f:
        train_grp = f.create_group("train")
        train_grp.create_dataset("low", data=train_low, compression="gzip")
        train_grp.create_dataset("high", data=train_high, compression="gzip")

        val_grp = f.create_group("val")
        val_grp.create_dataset("low", data=val_low, compression="gzip")
        val_grp.create_dataset("high", data=val_high, compression="gzip")

    file_size_mb = os.path.getsize(h5_output_path) / (1024 * 1024)
    print(f"\nSaved H5 file: {h5_output_path} ({file_size_mb:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Merge two ONEAT H5 datasets into a combined dataset.
Combines train/val splits from both sources.
"""

import argparse
import h5py
import numpy as np
from pathlib import Path


def get_dataset_info(h5_path):
    """Get info about an H5 dataset."""
    with h5py.File(h5_path, 'r') as f:
        info = {}
        for split in ['train', 'val']:
            if split in f:
                grp = f[split]
                info[split] = {
                    'images': grp['images'].shape[0] if 'images' in grp else 0,
                    'labels': grp['labels'].shape[0] if 'labels' in grp else 0,
                }
        return info


def merge_h5_datasets(h5_path_1, h5_path_2, output_path):
    """
    Merge two H5 datasets into one.

    Args:
        h5_path_1: Path to first H5 file
        h5_path_2: Path to second H5 file
        output_path: Path for merged output
    """
    print(f"Merging H5 datasets:")
    print(f"  Source 1: {h5_path_1}")
    print(f"  Source 2: {h5_path_2}")
    print(f"  Output:   {output_path}")

    # Get info about both datasets
    info1 = get_dataset_info(h5_path_1)
    info2 = get_dataset_info(h5_path_2)

    print(f"\nDataset 1: {info1}")
    print(f"Dataset 2: {info2}")

    with h5py.File(h5_path_1, 'r') as f1, \
         h5py.File(h5_path_2, 'r') as f2, \
         h5py.File(output_path, 'w') as out:

        for split in ['train', 'val']:
            print(f"\nProcessing {split} split...")

            out_grp = out.create_group(split)

            # Check which sources have this split
            has_split_1 = split in f1
            has_split_2 = split in f2

            if not has_split_1 and not has_split_2:
                print(f"  No {split} data in either source, skipping")
                continue

            # Get datasets from each source
            datasets_to_merge = {}

            for key in ['images', 'labels', 'seg']:
                arrays = []

                if has_split_1 and key in f1[split]:
                    arr1 = f1[split][key][:]
                    arrays.append(arr1)
                    print(f"  {key} from source 1: {arr1.shape}")

                if has_split_2 and key in f2[split]:
                    arr2 = f2[split][key][:]
                    arrays.append(arr2)
                    print(f"  {key} from source 2: {arr2.shape}")

                if arrays:
                    # Concatenate along first axis (samples)
                    merged = np.concatenate(arrays, axis=0)
                    datasets_to_merge[key] = merged
                    print(f"  {key} merged: {merged.shape}")

            # Write merged datasets
            for key, data in datasets_to_merge.items():
                out_grp.create_dataset(
                    key,
                    data=data,
                    maxshape=(None,) + data.shape[1:],
                    chunks=True,
                    compression='gzip'
                )
                print(f"  Wrote {key}: {data.shape}")

    # Print summary
    print(f"\n{'='*60}")
    print("Merge complete!")
    merged_info = get_dataset_info(output_path)
    print(f"Merged dataset: {merged_info}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Merge two ONEAT H5 datasets')
    parser.add_argument('h5_1', type=str, help='Path to first H5 file')
    parser.add_argument('h5_2', type=str, help='Path to second H5 file')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output path for merged H5')

    args = parser.parse_args()

    merge_h5_datasets(args.h5_1, args.h5_2, args.output)


if __name__ == "__main__":
    main()

"""
H5 Inspector for ONEAT Training Data
Inspects and visualizes ONEAT event classification H5 datasets
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path


def inspect_h5_file(dataset_path):
    """Inspect the raw H5 file structure."""
    print("\n" + "=" * 80)
    print("ONEAT H5 FILE STRUCTURE")
    print("=" * 80)
    print(f"File: {dataset_path}")
    print("-" * 80)

    with h5py.File(dataset_path, 'r') as f:
        print(f"Root keys: {list(f.keys())}")
        print("-" * 80)

        def explore_group(group, indent=0):
            """Recursively explore H5 groups."""
            prefix = "  " * indent
            for key in group.keys():
                try:
                    item = group[key]
                    if isinstance(item, h5py.Dataset):
                        print(f"{prefix}📊 {key}: shape={item.shape}, dtype={item.dtype}")
                        # Show value range for numeric data
                        if np.issubdtype(item.dtype, np.number) and item.size > 0:
                            sample = item[:min(100, len(item))]  # Sample first 100 items
                            print(f"{prefix}   └─ Range: [{np.min(sample):.3f}, {np.max(sample):.3f}]")
                    elif isinstance(item, h5py.Group):
                        print(f"{prefix}📁 {key}/")
                        explore_group(item, indent + 1)
                except Exception as e:
                    print(f"{prefix}⚠️ {key}: Error reading - {e}")

        explore_group(f)


def analyze_dataset_statistics(dataset_path, event_names=['normal', 'mitosis']):
    """Analyze dataset statistics."""
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    with h5py.File(dataset_path, 'r') as f:
        for split in ['train', 'val']:
            if split not in f:
                continue

            grp = f[split]
            print(f"\n{split.upper()} Split:")
            print("-" * 40)

            if 'images' in grp:
                n_samples = grp['images'].shape[0]
                print(f"Number of samples: {n_samples}")
                print(f"Image shape: {grp['images'].shape}")
                print(f"Segmentation shape: {grp['segmentations'].shape}")

                # Label distribution
                labels = grp['labels'][:]
                unique, counts = np.unique(labels, return_counts=True)
                print(f"\nLabel distribution:")
                for label, count in zip(unique, counts):
                    event_name = event_names[int(label)] if int(label) < len(event_names) else f"Class {label}"
                    percentage = (count / n_samples) * 100
                    print(f"  {event_name}: {count} samples ({percentage:.1f}%)")

                # Image statistics
                print(f"\nImage value statistics:")
                sample_images = grp['images'][:min(100, n_samples)]
                print(f"  Mean: {np.mean(sample_images):.3f}")
                print(f"  Std: {np.std(sample_images):.3f}")
                print(f"  Min: {np.min(sample_images):.3f}")
                print(f"  Max: {np.max(sample_images):.3f}")


def visualize_samples(dataset_path, n_samples=5, event_names=['normal', 'mitosis'], save_fig=None):
    """Visualize sample images from the dataset."""
    print("\n" + "=" * 80)
    print("VISUALIZING SAMPLES")
    print("=" * 80)

    with h5py.File(dataset_path, 'r') as f:
        if 'train' not in f:
            print("No training data found!")
            return

        grp = f['train']
        total_samples = grp['images'].shape[0]
        n_samples = min(n_samples, total_samples)

        # Randomly sample indices
        indices = np.random.choice(total_samples, n_samples, replace=False)

        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i, idx in enumerate(indices):
            image = grp['images'][idx]
            seg = grp['segmentations'][idx]
            label = grp['labels'][idx]
            event_name = event_names[int(label)] if int(label) < len(event_names) else f"Class {label}"

            # Get middle timepoint and Z-slice
            t_mid = image.shape[0] // 2
            z_mid = image.shape[1] // 2

            # Show raw image
            axes[i, 0].imshow(image[t_mid, z_mid], cmap='gray')
            axes[i, 0].set_title(f"Sample {idx}\n{event_name} - Raw (T={t_mid}, Z={z_mid})")
            axes[i, 0].axis('off')

            # Show segmentation
            axes[i, 1].imshow(seg[t_mid, z_mid], cmap='tab20')
            axes[i, 1].set_title(f"Segmentation (T={t_mid}, Z={z_mid})")
            axes[i, 1].axis('off')

            # Show temporal MIP
            temporal_mip = np.max(image[:, z_mid], axis=0)
            axes[i, 2].imshow(temporal_mip, cmap='gray')
            axes[i, 2].set_title(f"Temporal MIP (Z={z_mid})")
            axes[i, 2].axis('off')

        plt.tight_layout()

        if save_fig:
            plt.savefig(save_fig, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_fig}")
        else:
            plt.show()


def visualize_temporal_sequence(dataset_path, sample_idx=0, event_names=['normal', 'mitosis'], save_fig=None):
    """Visualize temporal sequence for a single sample."""
    print("\n" + "=" * 80)
    print("TEMPORAL SEQUENCE VISUALIZATION")
    print("=" * 80)

    with h5py.File(dataset_path, 'r') as f:
        if 'train' not in f:
            print("No training data found!")
            return

        grp = f['train']
        image = grp['images'][sample_idx]
        label = grp['labels'][sample_idx]
        event_name = event_names[int(label)] if int(label) < len(event_names) else f"Class {label}"

        n_timepoints = image.shape[0]
        z_mid = image.shape[1] // 2

        # Create figure with all timepoints
        fig, axes = plt.subplots(1, n_timepoints, figsize=(3 * n_timepoints, 3))
        if n_timepoints == 1:
            axes = [axes]

        for t in range(n_timepoints):
            axes[t].imshow(image[t, z_mid], cmap='gray')
            axes[t].set_title(f"T={t}")
            axes[t].axis('off')

        fig.suptitle(f"Sample {sample_idx} - {event_name} (Z={z_mid})", fontsize=14)
        plt.tight_layout()

        if save_fig:
            plt.savefig(save_fig, dpi=150, bbox_inches='tight')
            print(f"Saved temporal sequence to {save_fig}")
        else:
            plt.show()


def main():
    """Main inspection function."""
    # Default path - modify as needed
    dataset_path = "/lustre/fsn1/projects/rech/jsy/uzj81mi/oneat_training/oneat_kapoorlabs.h5"

    # Check if file exists
    if not Path(dataset_path).exists():
        print(f"Error: File not found at {dataset_path}")
        print("\nPlease provide the correct path to your H5 file.")
        return

    # Event names
    event_names = ['normal', 'mitosis']

    # Inspect file structure
    inspect_h5_file(dataset_path)

    # Analyze statistics
    analyze_dataset_statistics(dataset_path, event_names)

    # Visualize samples
    visualize_samples(dataset_path, n_samples=5, event_names=event_names)

    # Visualize temporal sequence
    visualize_temporal_sequence(dataset_path, sample_idx=0, event_names=event_names)


if __name__ == "__main__":
    main()

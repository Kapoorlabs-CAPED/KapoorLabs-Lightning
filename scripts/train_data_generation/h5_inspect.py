import numpy as np
import h5py


def inspect_h5_file(dataset_path):
    """Inspect the raw H5 file structure."""
    print("\n" + "=" * 80)
    print("H5 FILE STRUCTURE")
    print("=" * 80)

    with h5py.File(dataset_path, 'r') as f:
        print(f"Root keys: {list(f.keys())}")
        print("-" * 40)

        def explore_group(group, indent=0):
            """Recursively explore H5 groups."""
            prefix = "  " * indent
            for key in group.keys():
                try:
                    item = group[key]
                    if isinstance(item, h5py.Dataset):
                        print(f"{prefix}📊 {key}: shape={item.shape}, dtype={item.dtype}")
                    elif isinstance(item, h5py.Group):
                        print(f"{prefix}📁 {key}/")
                        explore_group(item, indent + 1)
                except Exception as e:
                    print(f"{prefix}⚠️ {key}: Error reading - {e}")

        explore_group(f)


def main():

    dataset_path = "/lustre/fsn1/projects/rech/jsy/uzj81mi/oneat_training/oneat_kapoorlabs.h5"

    inspect_h5_file(dataset_path)


if __name__ == "__main__":
    main()

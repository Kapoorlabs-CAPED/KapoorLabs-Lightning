import os
import json
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import h5py

from pathlib import Path
from omegaconf import OmegaConf
from skimage import measure

from bokeh.palettes import Category10
import itertools


logger = logging.getLogger(__name__)


def get_most_recent_file(file_path, file_pattern):
    ckpt_files = [file for file in os.listdir(file_path) if file.endswith(file_pattern)]

    if len(ckpt_files) > 0:
        # Sort by modification time (most recent first)
        ckpt_files_with_mtime = [
            (file, os.path.getmtime(os.path.join(file_path, file)))
            for file in ckpt_files
        ]
        sorted_ckpt_files = sorted(
            ckpt_files_with_mtime, key=lambda x: x[1], reverse=True
        )
        most_recent_ckpt = sorted_ckpt_files[0][0]
        return os.path.join(file_path, most_recent_ckpt)
    else:
        return None


def load_checkpoint_model(log_path: str):

    ckpt_path = get_most_recent_file(log_path, ".ckpt")

    return ckpt_path


def plot_npz_files_interactive(
    filepaths,
    unwanted_substrings=["gpu", "memory"],
    page_output_dir="metrics",
    save_plots=False,
    show_plots=True,
):
    all_data = {}
    Path(page_output_dir).mkdir(parents=True, exist_ok=True)

    # Load and merge data
    for filepath in filepaths:
        try:
            data = np.load(str(filepath), allow_pickle=True)
        except Exception as e:
            print(f"Skipping {filepath}: {e}")
            continue

        keys = sorted(data.files, key=lambda x: ("epoch" in x, x), reverse=True)
        for key in keys:
            if any(sub in key for sub in unwanted_substrings):
                continue
            data_values = data[key].tolist()
            if key not in all_data:
                all_data[key] = data_values
            else:
                all_data[key]["steps"].extend(data_values["steps"])
                all_data[key]["values"].extend(data_values["values"])

    # Prepare figure layout
    colors = itertools.cycle(Category10[10])
    grouped_keys = list(all_data.keys())
    n_cols = 4
    n_plots = len(grouped_keys)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    for i, key in enumerate(grouped_keys):
        values = all_data[key]
        df = pd.DataFrame.from_dict(values).sort_values("steps")
        color = next(colors)

        ax = axes[i]
        ax.plot(df["steps"].to_numpy(), df["values"].to_numpy(), label=key, color=color)
        ax.scatter(
            df["steps"].to_numpy(), df["values"].to_numpy(), s=4, color=color, alpha=0.3
        )
        ax.set_title(f"{key}")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_plots:
        output_path = Path(page_output_dir) / "metrics_all_in_one.png"
        fig.savefig(output_path, dpi=300)
        print(f"Saved all-in-one plot to: {output_path}")
        if show_plots:
            plt.show()

    if show_plots:
        plt.show()


def plot_npz_files(filepaths):
    all_data = {}
    for filepath in filepaths:
        try:
            data = np.load(str(filepath), allow_pickle=True)
        except pickle.UnpicklingError:
            # print(f"Error loading data from {filepath}. Skipping this file.")
            continue

        keys = data.files
        keys = sorted(keys, key=lambda x: ("epoch" in x, x), reverse=True)
        unwanted_substrings = ["step", "gpu", "memory"]
        for idx, key in enumerate(keys):
            if not any(substring in key for substring in unwanted_substrings):
                data_values = data[key].tolist()
                if key not in all_data:
                    all_data[key] = data_values
                else:
                    all_data[key]["steps"].extend(data_values["steps"])
                    all_data[key]["values"].extend(data_values["values"])
    for k, v in all_data.items():
        data_frame = pd.DataFrame.from_dict(all_data[k])
        sns.lineplot(x="steps", y="values", data=data_frame, label=k)
        plt.show()


def normalize_mi_ma(x, mi, ma, eps=1e-20, dtype=np.float32):
    x = x.astype(dtype)
    mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
    ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
    eps = dtype(eps) if np.isscalar(eps) else eps.astype(dtype, copy=False)

    x = (x - mi) / (ma - mi + eps)

    return x


def percentile_norm(x, pmin=1, pmax=99.8, axis=None, eps=1e-20, dtype=np.float32):
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, eps=eps, dtype=dtype)


def normalize_in_chunks(image, chunk_steps=50, pmin=1, pmax=99.8, dtype=np.float32):
    """
    Normalize a TZYX image in chunks along the T (time) dimension.

    Args:
        image (np.ndarray): The original TZYX image.
        chunk_steps (int): The number of timesteps to process at a time.
        pmin (float): The lower percentile for normalization.
        pmax (float): The upper percentile for normalization.
        dtype (np.dtype): The data type to cast the normalized image.

    Returns:
        np.ndarray: The normalized image with the same shape as the input.
    """
    # Get the shape of the original image (T, Z, Y, X)
    T = image.shape[0]

    # Create an empty array to hold the normalized image
    normalized_image = np.empty_like(image, dtype=dtype)

    # Process the image in chunks of `chunk_steps` along the T (time) axis
    for t in range(0, T, chunk_steps):
        # Determine the chunk slice, ensuring we don't go out of bounds
        t_end = min(t + chunk_steps, T)

        # Extract the chunk of timesteps to normalize
        chunk = image[t:t_end]

        # Normalize this chunk
        chunk_normalized = percentile_norm(chunk, pmin=pmin, pmax=pmax, dtype=dtype)

        # Replace the corresponding portion with the normalized chunk
        normalized_image[t:t_end] = chunk_normalized

    return normalized_image


def save_config_as_json(config, log_path):
    """Save resolved OmegaConf config as JSON to log_path"""
    config_dict = OmegaConf.to_container(config, resolve=True)

    config_file = Path(log_path) / "training_config.json"
    with open(config_file, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Config saved to: {config_file}")
    return config_dict


def create_event_dataset_h5(
    raw_images,
    seg_images,
    csv_files,
    event_names,
    h5_output_path,
    crop_size,
    train_split=0.95,
    batch_write_size=100,
    raw_files=None,
):

    # Build mapping from image basename to index
    if raw_files is None:
        raw_files = [f"image_{i}" for i in range(len(raw_images))]

    image_name_to_idx = {}
    for idx, raw_file in enumerate(raw_files):
        basename = os.path.basename(raw_file)
        image_name = os.path.splitext(basename)[0]
        image_name_to_idx[image_name] = idx

    event_data = []
    for csv_file in csv_files:
        csv_name = os.path.basename(csv_file)

        # Find which event and image this CSV corresponds to
        # CSV format: oneat_{event}_{image_name}.csv
        matched = False
        for event_idx, event_name in enumerate(event_names):
            if event_name in csv_name:
                # Extract image name from CSV: oneat_{event}_{image_name}.csv
                prefix = f"oneat_{event_name}_"
                if csv_name.startswith(prefix):
                    image_name = csv_name[len(prefix) :].replace(".csv", "")

                    # Find raw image index
                    if image_name in image_name_to_idx:
                        raw_idx = image_name_to_idx[image_name]
                        clicks_df = pd.read_csv(csv_file)

                        for _, row in clicks_df.iterrows():
                            event_data.append(
                                {
                                    "raw_idx": raw_idx,
                                    "csv_file": csv_file,
                                    "event_label": event_idx,
                                    "event_name": event_name,
                                    "time": row.get("t", row.get("time", 0)),
                                    "x": row.get("x", 0),
                                    "y": row.get("y", 0),
                                    "z": row.get("z", 0),
                                }
                            )
                        matched = True
                        break

        if not matched:
            print(f"Warning: Could not match CSV {csv_name} to any raw image")

    np.random.shuffle(event_data)
    train_size = int(len(event_data) * train_split)
    train_data = event_data[:train_size]
    val_data = event_data[train_size:]

    with h5py.File(h5_output_path, "w") as h5f:
        train_grp = h5f.create_group("train")
        val_grp = h5f.create_group("val")

        train_images_list = []
        train_segs_list = []
        train_labels_list = []

        val_images_list = []
        val_segs_list = []
        val_labels_list = []

        for i, event in enumerate(train_data):
            result = _extract_event_cube(
                raw_images[event["raw_idx"]],
                seg_images[event["raw_idx"]],
                event,
                crop_size,
            )
            if result is not None:
                crop_image, crop_seg, label = result
                train_images_list.append(crop_image)
                train_segs_list.append(crop_seg)
                train_labels_list.append(label)

                if len(train_images_list) >= batch_write_size:
                    _write_batch_to_h5(
                        train_grp, train_images_list, train_segs_list, train_labels_list
                    )
                    train_images_list = []
                    train_segs_list = []
                    train_labels_list = []

        if len(train_images_list) > 0:
            _write_batch_to_h5(
                train_grp, train_images_list, train_segs_list, train_labels_list
            )

        for event in val_data:
            result = _extract_event_cube(
                raw_images[event["raw_idx"]],
                seg_images[event["raw_idx"]],
                event,
                crop_size,
            )
            if result is not None:
                crop_image, crop_seg, label = result
                val_images_list.append(crop_image)
                val_segs_list.append(crop_seg)
                val_labels_list.append(label)

                if len(val_images_list) >= batch_write_size:
                    _write_batch_to_h5(
                        val_grp, val_images_list, val_segs_list, val_labels_list
                    )
                    val_images_list = []
                    val_segs_list = []
                    val_labels_list = []

        if len(val_images_list) > 0:
            _write_batch_to_h5(val_grp, val_images_list, val_segs_list, val_labels_list)


def _extract_event_cube(raw_image, seg_image, event, crop_size, num_classes=2):
    """
    Extract a cube around an event with fixed output shape and compute YOLO labels.

    Returns:
        crop_raw: (T, Z, Y, X) raw image crop
        crop_seg: (T, Z, Y, X) segmentation crop
        yolo_label: array of [x, y, z, h, w, d, conf] + [one-hot class (num_classes)]
    """
    sizex, sizey, sizez, t_minus, t_plus = crop_size
    time = int(event["time"])
    x = int(event["x"])
    y = int(event["y"])
    z = int(event["z"])
    event_label = int(event["event_label"])

    # Check temporal bounds
    t_start = time - t_minus
    t_end = time + t_plus + 1

    if t_start < 0 or t_end > raw_image.shape[0]:
        return None

    # Target output shape
    n_time = t_minus + t_plus + 1
    target_shape = (n_time, sizez, sizey, sizex)

    # Initialize output arrays with zeros (padding)
    crop_raw = np.zeros(target_shape, dtype=raw_image.dtype)
    crop_seg = np.zeros(target_shape, dtype=seg_image.dtype)

    # Calculate spatial bounds with clamping
    z_start = z - sizez // 2
    z_end = z_start + sizez
    y_start = y - sizey // 2
    y_end = y_start + sizey
    x_start = x - sizex // 2
    x_end = x_start + sizex

    # Clamp to image bounds
    z_start_src = max(0, z_start)
    z_end_src = min(raw_image.shape[1], z_end)
    y_start_src = max(0, y_start)
    y_end_src = min(raw_image.shape[2], y_end)
    x_start_src = max(0, x_start)
    x_end_src = min(raw_image.shape[3], x_end)

    # Calculate destination indices in padded array
    z_start_dst = z_start_src - z_start
    z_end_dst = z_start_dst + (z_end_src - z_start_src)
    y_start_dst = y_start_src - y_start
    y_end_dst = y_start_dst + (y_end_src - y_start_src)
    x_start_dst = x_start_src - x_start
    x_end_dst = x_start_dst + (x_end_src - x_start_src)

    # Copy data into padded arrays
    crop_raw[
        :, z_start_dst:z_end_dst, y_start_dst:y_end_dst, x_start_dst:x_end_dst
    ] = raw_image[
        t_start:t_end,
        z_start_src:z_end_src,
        y_start_src:y_end_src,
        x_start_src:x_end_src,
    ]

    crop_seg[
        :, z_start_dst:z_end_dst, y_start_dst:y_end_dst, x_start_dst:x_end_dst
    ] = seg_image[
        t_start:t_end,
        z_start_src:z_end_src,
        y_start_src:y_end_src,
        x_start_src:x_end_src,
    ]

    # Compute box_vector from segmentation at the event timepoint (middle of crop)
    mid_t = t_minus  # middle timepoint in the crop
    seg_at_event = crop_seg[mid_t]  # (Z, Y, X)

    # Find the cell label at the event location (center of crop)
    center_z, center_y, center_x = sizez // 2, sizey // 2, sizex // 2
    cell_label = seg_at_event[center_z, center_y, center_x]

    # If no cell at center, find nearest cell
    if cell_label == 0:
        nonzero_coords = np.argwhere(seg_at_event > 0)
        if len(nonzero_coords) > 0:
            center = np.array([center_z, center_y, center_x])
            distances = np.linalg.norm(nonzero_coords - center, axis=1)
            closest_idx = np.argmin(distances)
            closest_coord = nonzero_coords[closest_idx]
            cell_label = seg_at_event[
                closest_coord[0], closest_coord[1], closest_coord[2]
            ]

    # xyz is always 0.5 (center of crop)
    box_x, box_y, box_z = 0.5, 0.5, 0.5
    # t is always 0.5 (center of temporal window)
    box_t = 0.5

    # Compute hwd from regionprops
    if cell_label > 0:
        cell_mask = (seg_at_event == cell_label).astype(np.uint16)
        props = measure.regionprops(cell_mask)

        if len(props) > 0:
            prop = props[0]
            # bbox is (min_z, min_y, min_x, max_z, max_y, max_x)
            bbox = prop.bbox
            # Dimensions normalized to crop size
            box_d = (bbox[3] - bbox[0]) / sizez  # depth (z)
            box_h = (bbox[4] - bbox[1]) / sizey  # height (y)
            box_w = (bbox[5] - bbox[2]) / sizex  # width (x)
        else:
            box_d, box_h, box_w = 0.1, 0.1, 0.1
    else:
        box_d, box_h, box_w = 0.1, 0.1, 0.1

    # Confidence is always 1 for ground truth
    conf = 1.0

    # Create one-hot encoded class
    one_hot_class = np.zeros(num_classes, dtype=np.float32)
    one_hot_class[event_label] = 1.0

    # YOLO label format: [x, y, z, t, h, w, d, c] + [one-hot class]
    box_vector = np.array(
        [box_x, box_y, box_z, box_t, box_h, box_w, box_d, conf], dtype=np.float32
    )
    yolo_label = np.concatenate([box_vector, one_hot_class])

    return crop_raw, crop_seg, yolo_label


def _write_batch_to_h5(group, images, segs, labels):
    images_arr = np.array(images)
    segs_arr = np.array(segs)
    labels_arr = np.array(labels)

    if "images" not in group:
        group.create_dataset(
            "images",
            data=images_arr,
            maxshape=(None,) + images_arr.shape[1:],
            chunks=True,
            compression="gzip",
        )
        group.create_dataset(
            "segmentations",
            data=segs_arr,
            maxshape=(None,) + segs_arr.shape[1:],
            chunks=True,
            compression="gzip",
        )
        group.create_dataset(
            "labels",
            data=labels_arr,
            maxshape=(None,) + labels_arr.shape[1:],
            chunks=True,
        )
    else:
        old_size = group["images"].shape[0]
        new_size = old_size + len(images)
        group["images"].resize(new_size, axis=0)
        group["segmentations"].resize(new_size, axis=0)
        group["labels"].resize(new_size, axis=0)
        group["images"][old_size:new_size] = images_arr
        group["segmentations"][old_size:new_size] = segs_arr
        group["labels"][old_size:new_size] = labels_arr


__all__ = [
    "get_most_recent_file",
    "load_checkpoint_model",
    "plot_npz_files_interactive",
    "plot_npz_files",
    "save_config_as_json",
    "create_event_dataset_h5",
    "percentile_norm",
    "normalize_mi_ma",
]

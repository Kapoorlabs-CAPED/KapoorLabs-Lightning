import os
import json
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import torch
import h5py

from pathlib import Path
from omegaconf import OmegaConf
from scipy import spatial
from skimage import measure

from bokeh.palettes import Category10
import itertools


logger = logging.getLogger(__name__)





def get_most_recent_file(file_path, file_pattern):
    ckpt_files = [
        file for file in os.listdir(file_path) if file.endswith(file_pattern)
    ]

    if len(ckpt_files) > 0:
        # Sort by modification time (most recent first)
        ckpt_files_with_mtime = [
            (file, os.path.getmtime(os.path.join(file_path, file)))
            for file in ckpt_files
        ]
        sorted_ckpt_files = sorted(ckpt_files_with_mtime, key=lambda x: x[1], reverse=True)
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
    show_plots=True
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
        ax.scatter(df["steps"].to_numpy(), df["values"].to_numpy(), s=4, color=color, alpha=0.3)
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


def percentile_norm(
    x, pmin=1, pmax=99.8, axis=None, eps=1e-20, dtype=np.float32
):
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, eps=eps, dtype=dtype)


def normalize_in_chunks(
    image, chunk_steps=50, pmin=1, pmax=99.8, dtype=np.float32
):
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
        chunk_normalized = percentile_norm(
            chunk, pmin=pmin, pmax=pmax, dtype=dtype
        )

        # Replace the corresponding portion with the normalized chunk
        normalized_image[t:t_end] = chunk_normalized

    return normalized_image


def save_config_as_json(config, log_path):
    """Save resolved OmegaConf config as JSON to log_path"""
    config_dict = OmegaConf.to_container(config, resolve=True)

    config_file = Path(log_path) / "training_config.json"
    with open(config_file, 'w') as f:
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
                    image_name = csv_name[len(prefix):].replace('.csv', '')

                    # Find raw image index
                    if image_name in image_name_to_idx:
                        raw_idx = image_name_to_idx[image_name]
                        clicks_df = pd.read_csv(csv_file)

                        for _, row in clicks_df.iterrows():
                            event_data.append({
                                'raw_idx': raw_idx,
                                'csv_file': csv_file,
                                'event_label': event_idx,
                                'event_name': event_name,
                                'time': row.get('t', row.get('time', 0)),
                                'x': row.get('x', 0),
                                'y': row.get('y', 0),
                                'z': row.get('z', 0)
                            })
                        matched = True
                        break

        if not matched:
            print(f"Warning: Could not match CSV {csv_name} to any raw image")

    np.random.shuffle(event_data)
    train_size = int(len(event_data) * train_split)
    train_data = event_data[:train_size]
    val_data = event_data[train_size:]

    with h5py.File(h5_output_path, 'w') as h5f:
        train_grp = h5f.create_group('train')
        val_grp = h5f.create_group('val')

        train_images_list = []
        train_segs_list = []
        train_labels_list = []

        val_images_list = []
        val_segs_list = []
        val_labels_list = []

        for i, event in enumerate(train_data):
            result = _extract_event_cube(
                raw_images[event['raw_idx']],
                seg_images[event['raw_idx']],
                event,
                crop_size
            )
            if result is not None:
                crop_image, crop_seg, label = result
                train_images_list.append(crop_image)
                train_segs_list.append(crop_seg)
                train_labels_list.append(label)

                if len(train_images_list) >= batch_write_size:
                    _write_batch_to_h5(train_grp, train_images_list, train_segs_list, train_labels_list)
                    train_images_list = []
                    train_segs_list = []
                    train_labels_list = []

        if len(train_images_list) > 0:
            _write_batch_to_h5(train_grp, train_images_list, train_segs_list, train_labels_list)

        for event in val_data:
            result = _extract_event_cube(
                raw_images[event['raw_idx']],
                seg_images[event['raw_idx']],
                event,
                crop_size
            )
            if result is not None:
                crop_image, crop_seg, label = result
                val_images_list.append(crop_image)
                val_segs_list.append(crop_seg)
                val_labels_list.append(label)

                if len(val_images_list) >= batch_write_size:
                    _write_batch_to_h5(val_grp, val_images_list, val_segs_list, val_labels_list)
                    val_images_list = []
                    val_segs_list = []
                    val_labels_list = []

        if len(val_images_list) > 0:
            _write_batch_to_h5(val_grp, val_images_list, val_segs_list, val_labels_list)


def _extract_event_cube(raw_image, seg_image, event, crop_size):
    sizex, sizey, sizez, t_minus, t_plus = crop_size
    time = int(event['time'])
    x = int(event['x'])
    y = int(event['y'])
    z = int(event['z'])

    if time < t_minus or time + t_plus + 1 >= raw_image.shape[0]:
        return None

    starttime = time - t_minus
    endtime = time + t_plus + 1
    temporal_image = raw_image[starttime:endtime, :]
    temporal_seg = seg_image[starttime:endtime, :]

    currentsegimage = seg_image[time, :].astype('uint16')

    properties = measure.regionprops(currentsegimage)
    centroids = [prop.centroid for prop in properties]

    if len(centroids) == 0:
        return None

    tree = spatial.cKDTree(centroids)
    d_location = (z, y, x)
    distance_cell_mask, nearest_location = tree.query(d_location)

    if distance_cell_mask < 0.5 * sizex:
        z = int(centroids[nearest_location][0])
        y = int(centroids[nearest_location][1])
        x = int(centroids[nearest_location][2])
       
   

        if (x > sizex // 2 and y > sizey // 2 and z > sizez // 2 and
            x + sizex // 2 < raw_image.shape[3] and
            y + sizey // 2 < raw_image.shape[2] and
            z + sizez // 2 < raw_image.shape[1]):

            crop_xminus = x - sizex // 2
            crop_xplus = x + sizex // 2
            crop_yminus = y - sizey // 2
            crop_yplus = y + sizey // 2
            crop_zminus = z - sizez // 2
            crop_zplus = z + sizez // 2

            crop_image = temporal_image[
                :,
                crop_zminus:crop_zplus,
                crop_yminus:crop_yplus,
                crop_xminus:crop_xplus
            ]

            crop_seg = temporal_seg[
                :,
                crop_zminus:crop_zplus,
                crop_yminus:crop_yplus,
                crop_xminus:crop_xplus
            ]

            if crop_image.shape == (t_plus + t_minus + 1, sizez, sizey, sizex):
                label = event['event_label']
                return crop_image, crop_seg, label

    return None


def _write_batch_to_h5(group, images, segs, labels):
    if 'images' not in group:
        group.create_dataset(
            'images',
            data=np.array(images),
            maxshape=(None,) + images[0].shape,
            chunks=True,
            compression='gzip'
        )
        group.create_dataset(
            'segmentations',
            data=np.array(segs),
            maxshape=(None,) + segs[0].shape,
            chunks=True,
            compression='gzip'
        )
        group.create_dataset(
            'labels',
            data=np.array(labels),
            maxshape=(None,),
            chunks=True
        )
    else:
        old_size = group['images'].shape[0]
        new_size = old_size + len(images)
        group['images'].resize(new_size, axis=0)
        group['segmentations'].resize(new_size, axis=0)
        group['labels'].resize(new_size, axis=0)
        group['images'][old_size:new_size] = images
        group['segmentations'][old_size:new_size] = segs
        group['labels'][old_size:new_size] = labels



__all__ = [
    "get_most_recent_file",
    "load_checkpoint_model",
    "plot_npz_files_interactive",
    "plot_npz_files",
    "blockwise_causal_norm",
    "blockwise_sum",
    "save_config_as_json",
    "create_event_dataset_h5",
    "percentile_norm",
    "normalize_mi_ma"
]
#!/usr/bin/env python3
"""
Generate inception model training data for cell fate classification.

Takes TrackMate XML + segmentation image OR a pre-computed CSV,
creates sliding window tracklets, and saves as H5 training file.

Usage:
    python generate-cellfate-training-data.py
    python generate-cellfate-training-data.py train_data_paths=cellfate_datagen_default
    python generate-cellfate-training-data.py parameters.tracklet_length=30 parameters.stride=4
"""

import os
import numpy as np
import pandas as pd
import h5py
import hydra
from pathlib import Path
from hydra.core.config_store import ConfigStore
from sklearn.model_selection import train_test_split

from kapoorlabs_lightning.tracking.track_features import (
    SHAPE_FEATURES,
    DYNAMIC_FEATURES,
    SHAPE_DYNAMIC_FEATURES,
)
from kapoorlabs_lightning.tracking.track_prediction import create_training_tracklets
from scenario_generate_cellfate import CellFateDataGenClass


configstore = ConfigStore.instance()
configstore.store(name="CellFateDataGenClass", node=CellFateDataGenClass)


def load_dataframe_from_xml(xml_path, seg_path, mask_path, calibration_str):
    """Compute feature DataFrame from XML + optional segmentation."""
    from kapoorlabs_lightning.tracking.track_vectors import TrackVectors

    seg_image = None
    mask_image = None

    if seg_path:
        import tifffile
        seg_image = tifffile.imread(seg_path)

    if mask_path:
        import tifffile
        mask_image = tifffile.imread(mask_path)

    cal = None
    if calibration_str:
        cal = tuple(float(c) for c in calibration_str.split(","))

    tv = TrackVectors(xml_path, seg_image=seg_image, mask=mask_image, calibration=cal)
    return tv.to_dataframe()


def normalize_features(df, feature_columns):
    """Per-column z-score normalization of feature columns."""
    df_norm = df.copy()
    for col in feature_columns:
        if col in df_norm.columns:
            mean = df_norm[col].mean()
            std = df_norm[col].std()
            if std > 0:
                df_norm[col] = (df_norm[col] - mean) / std
            else:
                df_norm[col] = 0.0
    return df_norm


def parse_label_map(label_map_str):
    """Parse 'Name:0,Name:1,...' into {Name: 0, ...} dict."""
    if not label_map_str:
        return None
    mapping = {}
    for pair in label_map_str.split(","):
        name, idx = pair.strip().rsplit(":", 1)
        mapping[name.strip()] = int(idx)
    return mapping


@hydra.main(
    config_path="../conf", config_name="scenario_generate_cellfate"
)
def main(config: CellFateDataGenClass):
    # Parameters
    tracklet_length = config.parameters.tracklet_length
    stride = config.parameters.stride
    features = config.parameters.features
    do_normalize = config.parameters.normalize
    test_size = config.parameters.test_size
    seed = config.parameters.seed
    input_mode = config.parameters.input_mode
    label_column = config.parameters.label_column
    track_id_column = config.parameters.track_id_column
    t_min = config.parameters.t_min
    t_max = config.parameters.t_max
    min_track_duration = config.parameters.min_track_duration

    # Data paths
    csv_file = config.data_paths.csv_file
    xml_file = config.data_paths.xml_file
    seg_file = config.data_paths.seg_file
    mask_file = config.data_paths.mask_file
    calibration = config.data_paths.calibration
    label_map_str = config.data_paths.label_map
    output_h5 = config.data_paths.output_h5_file

    # Load DataFrame
    if input_mode == "csv":
        print(f"Loading CSV: {csv_file}")
        df = pd.read_csv(csv_file)
    else:
        print(f"Computing features from XML: {xml_file}")
        df = load_dataframe_from_xml(xml_file, seg_file, mask_file, calibration)

    print(f"Loaded {len(df)} rows, {df[track_id_column].nunique()} tracks")

    # Time filtering
    if t_min is not None:
        df = df[df["t"] >= t_min]
    if t_max is not None:
        df = df[df["t"] <= t_max]

    # Duration filtering
    if min_track_duration is not None:
        track_lengths = df.groupby(track_id_column).size()
        valid = track_lengths[track_lengths >= min_track_duration].index
        df = df[df[track_id_column].isin(valid)]

    # Select feature columns
    if features == "shape":
        feature_cols = [c for c in SHAPE_FEATURES if c in df.columns]
    elif features == "dynamic":
        feature_cols = [c for c in DYNAMIC_FEATURES if c in df.columns]
    else:
        feature_cols = [c for c in SHAPE_DYNAMIC_FEATURES if c in df.columns]

    print(f"Using {len(feature_cols)} features: {feature_cols}")

    # Map labels to integers
    label_map = parse_label_map(label_map_str)
    if label_map:
        df["label"] = df[label_column].map(label_map)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        print(f"Label mapping: {label_map}")
    elif label_column in df.columns:
        if df[label_column].dtype == object:
            unique_labels = sorted(df[label_column].dropna().unique())
            auto_map = {label: i for i, label in enumerate(unique_labels)}
            print(f"Auto-mapped labels: {auto_map}")
            df["label"] = df[label_column].map(auto_map)
            df = df.dropna(subset=["label"])
            df["label"] = df["label"].astype(int)
        else:
            df["label"] = df[label_column].astype(int)

    # Drop NaN features
    df = df.dropna(subset=feature_cols)

    # Normalize
    if do_normalize:
        print("Normalizing features...")
        df = normalize_features(df, feature_cols)

    # Create tracklets
    print(f"Creating tracklets (length={tracklet_length}, stride={stride})...")
    arrays, labels = create_training_tracklets(
        df,
        tracklet_length=tracklet_length,
        stride=stride,
        feature_columns=feature_cols,
        label_column="label",
        track_id_column=track_id_column,
    )

    if len(arrays) == 0:
        print("ERROR: No tracklets generated. Check track lengths vs tracklet_length.")
        return

    print(f"Generated {len(arrays)} tracklets, shape: {arrays.shape}")

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} samples ({100*cnt/len(labels):.1f}%)")

    # Train/val split
    train_arrays, val_arrays, train_labels, val_labels = train_test_split(
        arrays, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    print(f"Train: {len(train_arrays)}, Val: {len(val_arrays)}")

    # Save H5
    Path(output_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as hf:
        hf.create_dataset("train_arrays", data=train_arrays)
        hf.create_dataset("train_labels", data=train_labels)
        hf.create_dataset("val_arrays", data=val_arrays)
        hf.create_dataset("val_labels", data=val_labels)
        hf.attrs["tracklet_length"] = tracklet_length
        hf.attrs["stride"] = stride
        hf.attrs["features"] = features
        hf.attrs["feature_columns"] = ",".join(feature_cols)
        hf.attrs["num_classes"] = len(unique)

    print(f"Saved training data to: {output_h5}")


if __name__ == "__main__":
    main()

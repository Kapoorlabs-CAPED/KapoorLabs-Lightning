"""
Inception model prediction utilities for cell fate classification.

Provides functions for sampling tracklet sub-arrays, running model
inference, and aggregating predictions across tracklets.
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .track_features import SHAPE_DYNAMIC_FEATURES


def sample_subarrays(
    data: np.ndarray,
    tracklet_length: int,
    total_duration: int,
) -> List[np.ndarray]:
    """
    Sample all possible sub-arrays of fixed length from a track array.

    Args:
        data: (N, F) array of features over time.
        tracklet_length: Length of each sub-array.
        total_duration: Total track duration (number of timepoints).

    Returns:
        List of (tracklet_length, F) arrays.
    """
    max_start = total_duration - tracklet_length
    if max_start <= 0:
        return []

    start_indices = random.sample(range(max_start), max_start)
    subarrays = []
    for start in start_indices:
        end = start + tracklet_length
        if end <= total_duration:
            sub = data[start:end, :]
            if sub.shape[0] == tracklet_length:
                subarrays.append(sub)
    return subarrays


def make_prediction(
    input_data: np.ndarray,
    model: torch.nn.Module,
    device: str = "cpu",
) -> int:
    """
    Run a single sub-array through the model and return predicted class.

    Args:
        input_data: (T, F) array — one tracklet.
        model: Trained PyTorch model (e.g. InceptionNet).
        device: Device string.

    Returns:
        Predicted class index.
    """
    with torch.no_grad():
        tensor = (
            torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
        ).to(device)
        logits = model(tensor)
        probs = torch.softmax(logits[0], dim=0)
        _, predicted = torch.max(probs, 0)
    return predicted.item()


def predict_track(
    dataframe: pd.DataFrame,
    track_id: int,
    tracklet_length: int,
    class_map: Dict[int, str],
    model: torch.nn.Module,
    device: str = "cpu",
    feature_columns: Optional[List[str]] = None,
    track_id_column: str = "Track_ID",
    duration_column: str = "Track_Duration",
) -> str:
    """
    Predict cell type for an entire track using majority voting.

    Extracts feature sub-arrays from the track, runs each through
    the model, and returns the duration-weighted majority prediction.

    Args:
        dataframe: Track features DataFrame (from TrackVectors.to_dataframe()).
        track_id: Track ID to predict.
        tracklet_length: Required sub-array length.
        class_map: Dict mapping class index → class name.
        model: Trained model.
        device: Device string.
        feature_columns: Feature columns to use. Defaults to SHAPE_DYNAMIC_FEATURES.
        track_id_column: Column name for track ID.
        duration_column: Column name for track duration.

    Returns:
        Predicted class name string, or "UnClassified".
    """
    if feature_columns is None:
        feature_columns = SHAPE_DYNAMIC_FEATURES

    sub_df = dataframe[dataframe[track_id_column] == track_id]
    if len(sub_df) < tracklet_length:
        return "UnClassified"

    features = sub_df[feature_columns].values
    total_duration = len(features)

    subarrays = sample_subarrays(features, tracklet_length, total_duration)
    if not subarrays:
        return "UnClassified"

    predictions = []
    for sub_array in subarrays:
        pred_class = make_prediction(sub_array, model, device)
        predictions.append(pred_class)

    most_frequent = _get_most_frequent(predictions)
    if most_frequent is not None:
        return class_map.get(int(most_frequent), "UnClassified")
    return "UnClassified"


def predict_all_tracks(
    dataframe: pd.DataFrame,
    tracklet_length: int,
    class_map: Dict[int, str],
    model: torch.nn.Module,
    device: str = "cpu",
    feature_columns: Optional[List[str]] = None,
    track_id_column: str = "Track_ID",
    trackmate_track_id_column: str = "TrackMate_Track_ID",
    t_min: Optional[int] = None,
    t_max: Optional[int] = None,
) -> Dict[int, str]:
    """
    Predict cell type per TrackMate parent track via two-level aggregation.

    For each ``TrackMate_Track_ID``:
      1. Split into tracklets using ``track_id_column`` (sub-lineage IDs).
      2. Discard tracklets shorter than ``tracklet_length``.
      3. For each remaining tracklet, majority-vote across window
         predictions → a tracklet class.
      4. Duration-weighted vote across tracklets (weight = number of
         rows in the tracklet) → the final parent-track class.

    Matches NapaTrackMater's ``inception_model_prediction`` aggregation.

    Returns:
        Dict mapping ``TrackMate_Track_ID`` → predicted class name.
    """
    if feature_columns is None:
        feature_columns = SHAPE_DYNAMIC_FEATURES

    df = dataframe.copy()
    if t_min is not None:
        df = df[df["t"] > t_min]
    if t_max is not None:
        df = df[df["t"] <= t_max]

    if trackmate_track_id_column not in df.columns:
        # Fallback: treat each tracklet as its own parent track
        df[trackmate_track_id_column] = df[track_id_column]

    # Per-column z-score normalization — matches the preprocessing
    # applied before the training H5 was built (master_tracking.normalize_features).
    # Without this, inference on raw physical-unit features is catastrophically off.
    for col in feature_columns:
        if col not in df.columns:
            continue
        series = df[col].astype(float)
        mean = series.mean()
        std = series.std()
        if std and std > 0:
            df[col] = (series - mean) / std
        else:
            df[col] = 0.0

    predictions: Dict[int, str] = {}

    for parent_id, parent_df in df.groupby(trackmate_track_id_column):
        tracklet_votes: Counter = Counter()

        for tracklet_id, tracklet_df in parent_df.groupby(track_id_column):
            if len(tracklet_df) < tracklet_length:
                continue

            tracklet_df = tracklet_df.sort_values("t")
            features = tracklet_df[feature_columns].values
            subarrays = sample_subarrays(features, tracklet_length, len(features))
            if not subarrays:
                continue

            window_preds = [
                make_prediction(sub, model, device) for sub in subarrays
            ]
            tracklet_class = _get_most_frequent(window_preds)
            if tracklet_class is None:
                continue

            # Weight this tracklet's vote by its duration (row count)
            tracklet_votes[int(tracklet_class)] += len(tracklet_df)

        if not tracklet_votes:
            continue

        final_class_idx = tracklet_votes.most_common(1)[0][0]
        final_class = class_map.get(final_class_idx)
        if final_class:
            predictions[int(parent_id)] = final_class

    return predictions


def save_cell_type_predictions(
    dataframe: pd.DataFrame,
    class_map: Dict[int, str],
    predictions: Dict[int, str],
    save_dir: str,
    prefix: str = "",
    track_id_column: str = "Track_ID",
):
    """
    Save cell type predictions as per-class CSV files.

    Each CSV contains Track_ID, t, z, y, x for the first timepoint
    of each track predicted as that class.

    Args:
        dataframe: Track features DataFrame.
        class_map: Dict mapping class index → class name.
        predictions: Dict mapping track_id → predicted class name.
        save_dir: Directory to save CSV files.
        prefix: Optional prefix for filenames.
        track_id_column: Column for track ID.
    """
    os.makedirs(save_dir, exist_ok=True)

    for class_name in class_map.values():
        rows = []
        for track_id, pred_class in predictions.items():
            if pred_class == class_name:
                track_df = dataframe[dataframe[track_id_column] == track_id]
                if len(track_df) == 0:
                    continue
                first_row = track_df.loc[track_df["t"].idxmin()]
                rows.append(
                    {
                        track_id_column: track_id,
                        "t": first_row["t"],
                        "z": first_row["z"],
                        "y": first_row["y"],
                        "x": first_row["x"],
                    }
                )

        df_out = pd.DataFrame(rows)
        safe_name = class_name.lower().replace(" ", "_")
        filename = (
            f"{prefix}{safe_name}_predictions.csv"
            if prefix
            else f"{safe_name}_predictions.csv"
        )
        filepath = os.path.join(save_dir, filename)
        df_out.to_csv(filepath, index=False)
        print(f"Saved {len(df_out)} {class_name} predictions to {filepath}")


def create_training_tracklets(
    dataframe: pd.DataFrame,
    tracklet_length: int,
    stride: int = 1,
    feature_columns: Optional[List[str]] = None,
    label_column: str = "label",
    track_id_column: str = "Track_ID",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window training arrays from a labeled track DataFrame.

    Args:
        dataframe: Track features DataFrame with a label column.
        tracklet_length: Length of each training sub-array.
        stride: Step size for sliding window.
        feature_columns: Feature columns to extract. Defaults to SHAPE_DYNAMIC_FEATURES.
        label_column: Column containing class labels.
        track_id_column: Column for track ID.

    Returns:
        Tuple of (arrays, labels) where arrays is (N, tracklet_length, F)
        and labels is (N,).
    """
    if feature_columns is None:
        feature_columns = SHAPE_DYNAMIC_FEATURES

    all_arrays = []
    all_labels = []

    for track_id, group in dataframe.groupby(track_id_column):
        group = group.sort_values("t")
        features = group[feature_columns].values
        label = group[label_column].iloc[0]

        N = len(features)
        if N >= tracklet_length:
            num_windows = (N - tracklet_length) // stride + 1
            for i in range(num_windows):
                start = i * stride
                end = start + tracklet_length
                sub = features[start:end]
                all_arrays.append(sub)
                all_labels.append(label)

    return np.array(all_arrays), np.array(all_labels)


def _get_most_frequent(predictions: List[int]) -> Optional[int]:
    """Return the most frequent prediction, or None if empty."""
    if not predictions:
        return None
    counts = Counter(predictions)
    return counts.most_common(1)[0][0]

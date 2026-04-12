#!/usr/bin/env python3
"""
Run cell fate classification on tracked cells using a trained inception model.

Takes a feature DataFrame (CSV or computed from XML) and a trained checkpoint,
predicts cell type for each track, and saves per-class CSV annotations.

Usage:
    python predict-cellfate.py
    python predict-cellfate.py train_experiment_data_paths=cellfate_predict_default
    python predict-cellfate.py parameters.tracklet_length=30
"""

import os
import torch
import pandas as pd
import hydra
from pathlib import Path
from collections import Counter
from hydra.core.config_store import ConfigStore
import numpy as np
from kapoorlabs_lightning.cellfate_module import CellFateModule
from kapoorlabs_lightning.pytorch_models import InceptionNet, DenseNet, MitosisNet
from kapoorlabs_lightning.tracking.track_features import (
    SHAPE_FEATURES,
    DYNAMIC_FEATURES,
    SHAPE_DYNAMIC_FEATURES,
)
from kapoorlabs_lightning.tracking.track_prediction import (
    predict_all_tracks,
    save_cell_type_predictions,
    determine_transition_times,
    refine_transition_times,
)
from scenario_predict_cellfate_inception import CellFatePredictInceptionClass
from kapoorlabs_lightning.tracking.track_vectors import TrackVectors

configstore = ConfigStore.instance()
configstore.store(name="CellFatePredictInceptionClass", node=CellFatePredictInceptionClass)


def parse_class_map(class_map):
    """Parse class map from string '0:Basal,1:Radial' or dict {0: 'Basal'}."""
    if isinstance(class_map, str):
        mapping = {}
        for pair in class_map.split(","):
            idx, name = pair.strip().split(":", 1)
            mapping[int(idx)] = name.strip()
        return mapping
    return {int(k): str(v) for k, v in dict(class_map).items()}


def build_network(config: CellFatePredictInceptionClass):
    """Build model architecture from config parameters."""
    model_choice = config.parameters.model_choice
    num_classes = config.parameters.num_classes
    growth_rate = config.parameters.growth_rate
    block_config = tuple(config.parameters.block_config)
    num_init_features = config.parameters.num_init_features
    bottleneck_size = config.parameters.bottleneck_size
    kernel_size = config.parameters.kernel_size
    attn_heads = config.parameters.attn_heads
    seq_len = config.parameters.seq_len

    # Determine input_channels from feature set
    features = config.parameters.features
    if features == "shape":
        input_channels = len(SHAPE_FEATURES)
    elif features == "dynamic":
        input_channels = len(DYNAMIC_FEATURES)
    else:
        input_channels = len(SHAPE_DYNAMIC_FEATURES)

    if model_choice == "inception":
        return InceptionNet(
            input_channels=input_channels,
            num_classes=num_classes,
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=num_init_features,
            bottleneck_size=bottleneck_size,
            kernel_size=kernel_size,
            attn_heads=attn_heads,
            seq_len=seq_len,
        )
    elif model_choice == "densenet":
        return DenseNet(
            input_channels=input_channels,
            num_classes=num_classes,
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=num_init_features,
            bottleneck_size=bottleneck_size,
            kernel_size=kernel_size,
        )
    elif model_choice == "mitosisnet":
        return MitosisNet(
            input_channels=input_channels,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown model_choice: {model_choice}")


def load_dataframe(config: CellFatePredictInceptionClass):
    """Load feature DataFrame from CSV or XML."""
    if config.parameters.input_mode == "csv":
        print(f"Loading CSV: {config.experiment_data_paths.csv_file}")
        return pd.read_csv(config.experiment_data_paths.csv_file)
    else:
        
        xml_file = config.experiment_data_paths.xml_file
        seg_file = config.experiment_data_paths.seg_file
        mask_file = config.experiment_data_paths.mask_file
        xml_path = Path(xml_file)
        cache_csv = str(xml_path.with_name(f"features_{xml_path.stem}.csv"))
        reuse_cached = bool(
            getattr(config.experiment_data_paths, "reuse_cached_features", True)
        )
        if reuse_cached and os.path.exists(cache_csv):
            print(f"Loading cached features: {cache_csv}")
            return pd.read_csv(cache_csv)
        voxel_size_xyz = getattr(config.experiment_data_paths, "voxel_size_xyz", None)
        variable_t_calibration = getattr(
            config.experiment_data_paths, "variable_t_calibration", None
        )
        if variable_t_calibration is not None:
            variable_t_calibration = {
                int(k): float(v) for k, v in dict(variable_t_calibration).items()
            }

        print(f"Computing features from XML: {xml_file}")
        seg_image = None
        mask_image = None
        if seg_file:
            import tifffile
            seg_image = tifffile.imread(seg_file)
        if mask_file:
            import tifffile
            mask_image = tifffile.imread(mask_file)
        cal = None
        if voxel_size_xyz is not None:
            vx, vy, vz = (float(v) for v in voxel_size_xyz)
            cal = (vz, vy, vx)
        is_master_xml = getattr(
            config.experiment_data_paths, "is_master_xml", None
        )
        tv = TrackVectors(
            xml_file,
            seg_image=seg_image,
            mask=mask_image,
            calibration=cal,
            variable_t_calibration=variable_t_calibration,
            is_master_xml=is_master_xml,
        )
        df = tv.to_dataframe()
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_csv, index=False)
        print(f"Cached features to: {cache_csv}")
        print(f"Feature DataFrame preview ({len(df)} rows, {df.shape[1]} cols):")
        first = df.iloc[0]
        print(", ".join(f"{col}={first[col]}" for col in df.columns))
        return df


@hydra.main(
    config_path="../conf", config_name="scenario_predict_cellfate_inception", version_base='1.3'
)
def main(config: CellFatePredictInceptionClass):
    accelerator = config.parameters.accelerator
    device = "cuda" if accelerator == "cuda" and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build model
    network = build_network(config)

    # Load checkpoint (resolve directory to latest .ckpt if needed)
    checkpoint_path = config.experiment_data_paths.checkpoint_path
    if os.path.isdir(checkpoint_path):
        from kapoorlabs_lightning.utils import load_checkpoint_model
        resolved = load_checkpoint_model(checkpoint_path)
        if resolved is None:
            raise FileNotFoundError(f"No .ckpt file found in {checkpoint_path}")
        checkpoint_path = resolved
    print(f"Loading checkpoint: {checkpoint_path}")
    model = CellFateModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        network=network,
        weights_only=False,
        strict=False,
    )
    model.eval()
    model.to(device)

    # Load data
    df = load_dataframe(config)
    track_id_column = config.parameters.track_id_column
    print(f"Loaded {len(df)} rows, {df[track_id_column].nunique()} tracks")

    # Restrict to user-chosen [start, end] frame range (inclusive, pixel/frame units).
    # Span must be > 50. Set to null to disable.
    time_window = getattr(config.parameters, "time_window", None)
    window_tag = ""
    full_df = df  # default: no truncation, so full == truncated
    if time_window is not None:
        tw = list(time_window)
        if len(tw) != 2:
            raise ValueError(
                f"time_window must be [start, end]; got {tw}"
            )
        t_start, t_end = int(tw[0]), int(tw[1])
        if t_end == -1:
            t_end = int(df["t"].max())
        if t_end - t_start + 1 <= 50:
            raise ValueError(
                f"time_window span must be > 50; got [{t_start}, {t_end}] "
                f"(span={t_end - t_start + 1}). Set to null to disable."
            )
        before = len(df)
        full_df = df  # keep full-range df for transition-time analysis
        df = df[(df["t"] >= t_start) & (df["t"] <= t_end)].reset_index(drop=True)
        window_tag = f"t{t_start}-{t_end}_"
        print(
            f"Time window: t in [{t_start}, {t_end}], "
            f"{before} -> {len(df)} rows, {df[track_id_column].nunique()} tracks"
        )

    # Parse class map
    class_map = parse_class_map(config.experiment_data_paths.class_map)
    print(f"Class map: {class_map}")

    # Select features
    features = config.parameters.features
    if features == "shape":
        feature_cols = [c for c in SHAPE_FEATURES if c in df.columns]
    elif features == "dynamic":
        feature_cols = [c for c in DYNAMIC_FEATURES if c in df.columns]
    else:
        feature_cols = [c for c in SHAPE_DYNAMIC_FEATURES if c in df.columns]

    tracklet_length = config.parameters.tracklet_length
    print(f"Using {len(feature_cols)} features, tracklet_length={tracklet_length}")

    # Predictions are keyed on TrackMate_Track_ID (parent track).
    parent_id_column = "TrackMate_Track_ID"

    # Run predictions (two-level aggregation: parent -> tracklet -> window)
    predictions = predict_all_tracks(
        df,
        tracklet_length=tracklet_length,
        class_map=class_map,
        model=model.network,
        device=device,
        feature_columns=feature_cols,
        track_id_column=track_id_column,
        trackmate_track_id_column=parent_id_column,
        t_min=config.parameters.t_min,
        t_max=config.parameters.t_max,
    )

    print(f"Predicted {len(predictions)} parent tracks")

    # Count per class
    class_counts = Counter(predictions.values())
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls}: {cnt} tracks")

    # Save predictions
    output_dir = config.experiment_data_paths.output_dir
    base_prefix = str(config.experiment_data_paths.output_prefix)
    prefix = f"{base_prefix}{window_tag}"

    save_cell_type_predictions(
        df, class_map, predictions, output_dir,
        prefix=prefix, track_id_column=parent_id_column,
    )

    # Save combined predictions CSV
    full_csv_name = f"{prefix}all_predictions.csv" if prefix else "all_predictions.csv"
    full_csv = os.path.join(output_dir, full_csv_name)
    pred_df = pd.DataFrame([
        {parent_id_column: tid, "Predicted_Class": cls}
        for tid, cls in predictions.items()
    ])
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(full_csv, index=False)
    print(f"Saved all predictions to: {full_csv}")

    # Transition-time determination via sliding windows (on by default)
    if bool(getattr(config.parameters, "transition_time_determination", True)):
        span = int(getattr(config.parameters, "transition_window_span", 50))
        stride = int(getattr(config.parameters, "transition_window_stride", 25))
        conf_df, transitions = determine_transition_times(
            full_df,
            tracklet_length=tracklet_length,
            class_map=class_map,
            model=model.network,
            device=device,
            feature_columns=feature_cols,
            track_id_column=track_id_column,
            trackmate_track_id_column=parent_id_column,
            window_span=span,
            window_stride=stride,
        )
        conf_csv = os.path.join(output_dir, f"{base_prefix}transition_confidence.csv")
        conf_df.to_csv(conf_csv, index=False)
        print(f"Saved transition-confidence table to: {conf_csv}")
        print("Peak-confidence window per class (coarse):")
        for cls, (ws, we) in transitions.items():
            print(f"  {cls}: t in [{ws}, {we}]")

        refine_levels = int(getattr(config.parameters, "transition_refine_levels", 2))
        refine_factor = int(getattr(config.parameters, "transition_refine_factor", 2))
        if refine_levels > 0 and transitions:
            refined, _ = refine_transition_times(
                full_df,
                tracklet_length=tracklet_length,
                class_map=class_map,
                model=model.network,
                coarse_transitions=transitions,
                device=device,
                feature_columns=feature_cols,
                track_id_column=track_id_column,
                trackmate_track_id_column=parent_id_column,
                initial_span=span,
                initial_stride=stride,
                levels=refine_levels,
                refine_factor=refine_factor,
            )
            print("Refined peak-confidence window per class:")
            for cls, (ws, we) in refined.items():
                print(f"  {cls}: t in [{ws}, {we}]")
            refined_rows = [
                {"class": cls, "window_start": ws, "window_end": we}
                for cls, (ws, we) in refined.items()
            ]
            refined_csv = os.path.join(output_dir, f"{base_prefix}transition_refined.csv")
            pd.DataFrame(refined_rows).to_csv(refined_csv, index=False)
            print(f"Saved refined transitions to: {refined_csv}")
            transitions = refined
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            centers = (conf_df["window_start"] + conf_df["window_end"]) / 2.0
            fig, ax = plt.subplots(figsize=(7, 4))
            for cls in class_map.values():
                col = f"conf_{cls}"
                if col in conf_df.columns:
                    ax.plot(centers, conf_df[col], marker="o", label=cls)
            for cls, (ws, we) in transitions.items():
                ax.axvline((ws + we) / 2.0, linestyle="--", alpha=0.3)
            ax.set_xlabel("Window center (t, frames)")
            ax.set_ylabel("Confidence (fraction agreeing with consensus)")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"Per-class transition confidence (span={span}, stride={stride})")
            ax.legend()
            fig.tight_layout()
            png = os.path.join(output_dir, f"{base_prefix}transition_confidence.png")
            fig.savefig(png, dpi=150)
            plt.close(fig)
            print(f"Saved transition-confidence plot to: {png}")
        except Exception as e:
            print(f"Transition plot skipped: {e}")

    # Optional GT comparison -> confusion matrix (GT points mapped to parent track)
    evaluate_against_gt(
        df, predictions, class_map, config.experiment_data_paths,
        output_dir, prefix, parent_id_column,
    )


def _gt_track_ids(df, gt_csv, track_id_column):
    """Map each GT (T,Z,Y,X) to a Track_ID via nearest spot at that frame."""
    from scipy.spatial import cKDTree
    gt = pd.read_csv(gt_csv)
    cols = {c.lower(): c for c in gt.columns}
    T = gt[cols.get("t", "T")].astype(int).values
    Z = gt[cols.get("z", "Z")].astype(float).values
    Y = gt[cols.get("y", "Y")].astype(float).values
    X = gt[cols.get("x", "X")].astype(float).values

    track_ids = []
    for t_val in np.unique(T):
        frame_df = df[df["t"] == int(t_val)]
        if frame_df.empty:
            continue
        tree = cKDTree(frame_df[["z", "y", "x"]].values)
        mask = T == t_val
        query = np.column_stack([Z[mask], Y[mask], X[mask]])
        _, idx = tree.query(query)
        track_ids.extend(frame_df.iloc[idx][track_id_column].tolist())
    return track_ids


def evaluate_against_gt(df, predictions, class_map, paths,
                        output_dir, prefix, track_id_column):
    """Build confusion matrix from GT csv files, if provided."""
    gt_map = {
        "Basal": getattr(paths, "basal_gt_annotations", None),
        "Goblet": getattr(paths, "goblet_gt_annotations", None),
        "Radial": getattr(paths, "radially_intercalating_gt_annotations", None),
    }
    gt_map = {k: v for k, v in gt_map.items() if v and os.path.exists(v)}
    if not gt_map:
        print("No GT annotation files found; skipping confusion matrix.")
        return

    track_true = {}  # track_id -> gt class
    conflicts = set()
    for cls, csv_path in gt_map.items():
        tids = _gt_track_ids(df, csv_path, track_id_column)
        print(f"GT {cls}: {len(tids)} points -> {len(set(tids))} unique tracks")
        for tid in tids:
            if tid in track_true and track_true[tid] != cls:
                conflicts.add(tid)
                continue
            track_true[tid] = cls
    if conflicts:
        print(f"Discarding {len(conflicts)} tracks with conflicting GT labels: "
              f"{sorted(conflicts)[:10]}{'...' if len(conflicts) > 10 else ''}")
        for tid in conflicts:
            track_true.pop(tid, None)

    # Pair with predictions (GT tracks absent from predictions are discarded)
    labels = list(class_map.values())
    y_true, y_pred = [], []
    dropped = 0
    for tid, true_cls in track_true.items():
        if tid in predictions:
            y_true.append(true_cls)
            y_pred.append(predictions[tid])
        else:
            dropped += 1
    if dropped:
        print(f"Discarded {dropped} GT tracks not present in predictions "
              f"(likely filtered by time_window / tracklet_length).")
    if not y_true:
        print("No overlap between GT tracks and predictions.")
        return

    n = len(labels)
    idx = {c: i for i, c in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1

    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in labels],
                         columns=[f"pred_{c}" for c in labels])
    print(f"\nConfusion matrix (rows=true, cols=pred) on {len(y_true)} tracks:")
    print(cm_df.to_string())
    # Accuracy: fraction of GT-labeled tracks (that have predictions)
    # whose prediction matches GT. Prediction-only tracks without GT
    # do not enter this denominator.
    correct = int(np.trace(cm))
    total_gt_with_pred = len(y_true)
    print(
        f"Accuracy (GT tracks correctly predicted): "
        f"{correct}/{total_gt_with_pred} = {correct/total_gt_with_pred:.4f}"
    )

    cm_csv = os.path.join(output_dir, f"{prefix}confusion_matrix.csv")
    cm_df.to_csv(cm_csv)
    print(f"Saved confusion matrix to: {cm_csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        png = os.path.join(output_dir, f"{prefix}confusion_matrix.png")
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"Saved confusion matrix plot to: {png}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()

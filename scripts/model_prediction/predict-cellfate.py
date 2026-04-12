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
)
from scenario_predict_cellfate_inception import CellFatePredictInceptionClass
from kapoorlabs_lightning.tracking.track_vectors import TrackVectors

configstore = ConfigStore.instance()
configstore.store(name="CellFatePredictInceptionClass", node=CellFatePredictInceptionClass)


def parse_class_map(class_map_str):
    """Parse '0:Basal,1:Radial,2:Goblet' into {0: 'Basal', ...}."""
    mapping = {}
    for pair in class_map_str.split(","):
        idx, name = pair.strip().split(":", 1)
        mapping[int(idx)] = name.strip()
    return mapping


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
        tv = TrackVectors(
            xml_file,
            seg_image=seg_image,
            mask=mask_image,
            calibration=cal,
            variable_t_calibration=variable_t_calibration,
        )
        return tv.to_dataframe()


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

    # Run predictions
    predictions = predict_all_tracks(
        df,
        tracklet_length=tracklet_length,
        class_map=class_map,
        model=model.network,
        device=device,
        feature_columns=feature_cols,
        track_id_column=track_id_column,
        t_min=config.parameters.t_min,
        t_max=config.parameters.t_max,
    )

    print(f"Predicted {len(predictions)} tracks")

    # Count per class
    class_counts = Counter(predictions.values())
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls}: {cnt} tracks")

    # Save predictions
    output_dir = config.experiment_data_paths.output_dir
    prefix = config.experiment_data_paths.output_prefix

    save_cell_type_predictions(
        df, class_map, predictions, output_dir,
        prefix=prefix, track_id_column=track_id_column,
    )

    # Save combined predictions CSV
    full_csv_name = f"{prefix}all_predictions.csv" if prefix else "all_predictions.csv"
    full_csv = os.path.join(output_dir, full_csv_name)
    pred_df = pd.DataFrame([
        {"Track_ID": tid, "Predicted_Class": cls}
        for tid, cls in predictions.items()
    ])
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(full_csv, index=False)
    print(f"Saved all predictions to: {full_csv}")


if __name__ == "__main__":
    main()

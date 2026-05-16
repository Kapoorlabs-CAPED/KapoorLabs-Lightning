import os
from glob import glob
from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra.core.config_store import ConfigStore
from lightning import Trainer
from tifffile import imread, imwrite
from torch.utils.data import DataLoader

from kapoorlabs_lightning.oneat_module import OneatActionModule
from kapoorlabs_lightning.oneat_prediction_dataset import OneatPredictionDataset
from kapoorlabs_lightning.oneat_presets import OneatEvalPreset
from kapoorlabs_lightning.pytorch_callbacks import EventCountProgressBar
from kapoorlabs_lightning.pytorch_models import DenseVollNet
from kapoorlabs_lightning.utils import load_checkpoint_model
from scenario_predict_oneat import OneatPredictClass
from _arch_loader import load_arch_from_training_config


configstore = ConfigStore.instance()
configstore.store(name="OneatPredictClass", node=OneatPredictClass)


@hydra.main(config_path="../conf", config_name="scenario_predict_oneat", version_base='1.3')
def main(config: OneatPredictClass):
    # JSON next to the ckpt wins over the Hydra parameter yaml — the
    # checkpoint may have been trained with a different patch / event /
    # arch config than the current `parameters/oneat.yaml`.
    log_path = config.train_data_paths.log_path
    json_params = load_arch_from_training_config(log_path)
    if json_params:
        print(f"Loaded arch from {log_path}/training_config.json")

    p = config.parameters

    # Inference scaffolding
    num_classes = json_params.get("num_classes", p.num_classes)
    devices = p.devices
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    # Model architecture parameters
    imagex = json_params.get("imagex", p.imagex)
    imagey = json_params.get("imagey", p.imagey)
    imagez = json_params.get("imagez", p.imagez)
    size_tminus = json_params.get("size_tminus", p.size_tminus)
    size_tplus = json_params.get("size_tplus", p.size_tplus)

    # Normalization parameters
    normalizeimage = p.normalizeimage
    pmin = p.pmin
    pmax = p.pmax

    # NMS and threshold parameters
    nms_iou_threshold = p.nms_iou_threshold
    event_threshold = p.event_threshold
    batch_size_predict = p.batch_size_predict

    # Event parameters
    event_names = json_params.get("event_name", p.event_name)

    # Data paths
    base_data_dir = config.experiment_data_paths.base_data_dir
    raw_timelapses_dir = os.path.join(
        base_data_dir, config.experiment_data_paths.raw_timelapses
    )
    seg_timelapses_dir = os.path.join(
        base_data_dir, config.experiment_data_paths.seg_timelapses
    )
    predictions_dir = os.path.join(
        base_data_dir, config.experiment_data_paths.oneat_predictions
    )

    # Test prediction: crop a small ROI from center of each timelapse
    test_pred = config.parameters.test_pred
    test_roi_xy = config.parameters.test_roi_xy

    if test_pred:
        print(
            f"\nTest prediction mode: cropping {test_roi_xy}x{test_roi_xy} XY ROI from center"
        )
        raw_files_full = sorted(
            glob(os.path.join(raw_timelapses_dir, config.parameters.file_type))
        )
        for raw_file in raw_files_full:
            basename = os.path.basename(raw_file)
            seg_file = os.path.join(seg_timelapses_dir, basename)
            if not os.path.exists(seg_file):
                continue

            raw_img = imread(raw_file)
            seg_img = imread(seg_file)

            # Image is TZYX — crop XY around center
            _, _, h, w = raw_img.shape if raw_img.ndim == 4 else (1, *raw_img.shape)
            cy, cx = h // 2, w // 2
            half = test_roi_xy // 2
            y0 = max(0, cy - half)
            y1 = min(h, cy + half)
            x0 = max(0, cx - half)
            x1 = min(w, cx + half)

            if raw_img.ndim == 4:
                raw_crop = raw_img[:, :, y0:y1, x0:x1]
                seg_crop = seg_img[:, :, y0:y1, x0:x1]
            else:
                raw_crop = raw_img[:, y0:y1, x0:x1]
                seg_crop = seg_img[:, y0:y1, x0:x1]

            test_name = "test_dataset.tif"
            raw_test_path = os.path.join(raw_timelapses_dir, test_name)
            seg_test_path = os.path.join(seg_timelapses_dir, test_name)
            imwrite(raw_test_path, raw_crop)
            imwrite(seg_test_path, seg_crop)
            print(f"Saved test crop: {raw_crop.shape} -> {raw_test_path}")
            print(f"Saved test crop: {seg_crop.shape} -> {seg_test_path}")
            # Only crop the first file for test
            break

    # Model checkpoint path from config
    ckpt_path = load_checkpoint_model(log_path)

    if ckpt_path is None:
        raise ValueError(f"No checkpoint found in {log_path}")

    print(f"Loading model from: {ckpt_path}")

    # Create predictions directory
    Path(predictions_dir).mkdir(exist_ok=True, parents=True)

    # Build model architecture (needed for load_from_checkpoint) — JSON wins.
    startfilter = json_params.get("startfilter", p.startfilter)
    start_kernel = json_params.get("start_kernel", p.start_kernel)
    mid_kernel = json_params.get("mid_kernel", p.mid_kernel)
    depth = json_params.get("depth", p.depth)
    growth_rate = json_params.get("growth_rate", p.growth_rate)
    pool_first = json_params.get("pool_first", p.pool_first)
    event_position_label = json_params.get(
        "event_position_label", p.event_position_label
    )

    n_time = size_tminus + size_tplus + 1
    input_shape = (n_time, imagez, imagey, imagex)
    box_vector = len(event_position_label)

    network = DenseVollNet(
        input_shape,
        num_classes,
        box_vector,
        start_kernel=start_kernel,
        mid_kernel=mid_kernel,
        startfilter=startfilter,
        depth=depth,
        growth_rate=growth_rate,
        pool_first=pool_first,
    )

    # Create eval transforms (same as validation during training)
    eval_transforms = OneatEvalPreset(
        percentile_norm=True,
        pmin=pmin,
        pmax=pmax,
    )

    # Load model from checkpoint using Lightning
    lightning_model = OneatActionModule.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
        network=network,
        eval_transforms=eval_transforms,
        imagex=imagex,
        imagey=imagey,
        imagez=imagez,
        size_tminus=size_tminus,
        size_tplus=size_tplus,
        event_names=event_names,
        num_classes=num_classes,
        event_threshold=event_threshold,
        nms_iou_threshold=nms_iou_threshold,
        batch_size_predict=batch_size_predict,
    )

    # Get raw tif files — only test_dataset.tif if test_pred mode
    if test_pred:
        raw_files = [os.path.join(raw_timelapses_dir, "test_dataset.tif")]
    else:
        raw_files = sorted(
            glob(os.path.join(raw_timelapses_dir, config.parameters.file_type))
        )

    print(f"Found {len(raw_files)} raw timelapse files")
    print(
        f"Event threshold: {event_threshold}, NMS IoU threshold: {nms_iou_threshold}, "
        f"Batch size (predict): {batch_size_predict}"
    )

    # Create Lightning Trainer for prediction
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        callbacks=[EventCountProgressBar()],
    )

    for raw_file in raw_files:
        print(f"\nProcessing: {os.path.basename(raw_file)}")

        # Find corresponding seg file
        raw_basename = os.path.basename(raw_file)
        seg_file = os.path.join(seg_timelapses_dir, raw_basename)

        if not os.path.exists(seg_file):
            print(f"Warning: Segmentation file not found: {seg_file}, skipping...")
            continue

        # Create prediction dataset
        pred_dataset = OneatPredictionDataset(
            raw_file=raw_file,
            seg_file=seg_file,
            size_tminus=size_tminus,
            size_tplus=size_tplus,
            normalize=normalizeimage,
            pmin=pmin,
            pmax=pmax,
            chunk_steps=50,
        )

        # Create dataloader (batch_size=1: one timepoint per batch,
        # all cells within that timepoint are batched inside predict_step)
        pred_dataloader = DataLoader(
            pred_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        # Run prediction using Lightning Trainer
        # NMS is applied online inside predict_step
        print("Running predictions...")
        predictions = trainer.predict(lightning_model, pred_dataloader)
        print()  # newline after progress bar

        # Flatten predictions (each batch returns a list of detections)
        all_detections = []
        for batch_detections in predictions:
            all_detections.extend(batch_detections)

        print(f"Total detections (post-threshold, post-NMS): {len(all_detections)}")

        # Save predictions to CSV
        if len(all_detections) > 0:
            df = pd.DataFrame(all_detections)

            # Group by event type and save separate CSV files
            for event_name in df["event_name"].unique():
                event_df = df[df["event_name"] == event_name]
                output_df = event_df[
                    ["time", "z", "y", "x", "score", "size", "h", "w", "d"]
                ].rename(columns={"time": "t"})
                csv_filename = (
                    f"oneat_{event_name}_{os.path.splitext(raw_basename)[0]}.csv"
                )
                csv_path = os.path.join(predictions_dir, csv_filename)
                output_df.to_csv(csv_path, index=False)
                print(f"Saved {len(event_df)} {event_name} detections to: {csv_path}")
        else:
            print("No events detected")

    print("\nPrediction complete!")


if __name__ == "__main__":
    main()

import os
from glob import glob
from pathlib import Path

import hydra
import numpy as np
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


configstore = ConfigStore.instance()
configstore.store(name="OneatPredictClass", node=OneatPredictClass)


@hydra.main(
    config_path="../conf", config_name="scenario_predict_oneat"
)
def main(config: OneatPredictClass):
    # Extract parameters
    num_classes = config.parameters.num_classes
    devices = config.parameters.devices
    accelerator = config.parameters.accelerator

    # Model architecture parameters
    imagex = config.parameters.imagex
    imagey = config.parameters.imagey
    imagez = config.parameters.imagez
    size_tminus = config.parameters.size_tminus
    size_tplus = config.parameters.size_tplus

    # Normalization parameters
    normalizeimage = config.parameters.normalizeimage
    pmin = config.parameters.pmin
    pmax = config.parameters.pmax

    # NMS and threshold parameters
    nms_space = config.parameters.nms_space
    nms_time = config.parameters.nms_time
    event_threshold = config.parameters.event_threshold
    batch_size_predict = config.parameters.batch_size_predict

    # Event parameters
    event_names = config.parameters.event_name

    # Data paths
    base_data_dir = config.experiment_data_paths.base_data_dir
    raw_timelapses_dir = os.path.join(base_data_dir, config.experiment_data_paths.raw_timelapses)
    seg_timelapses_dir = os.path.join(base_data_dir, config.experiment_data_paths.seg_timelapses)
    predictions_dir = os.path.join(base_data_dir, config.experiment_data_paths.oneat_predictions)

    # Test prediction: crop a small ROI from center of each timelapse
    test_pred = config.parameters.test_pred
    test_roi_xy = config.parameters.test_roi_xy

    if test_pred:
        print(f"\nTest prediction mode: cropping {test_roi_xy}x{test_roi_xy} XY ROI from center")
        raw_files_full = sorted(glob(os.path.join(raw_timelapses_dir, config.parameters.file_type)))
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
    log_path = config.train_data_paths.log_path
    ckpt_path = load_checkpoint_model(log_path)

    if ckpt_path is None:
        raise ValueError(f"No checkpoint found in {log_path}")

    print(f"Loading model from: {ckpt_path}")

    # Create predictions directory
    Path(predictions_dir).mkdir(exist_ok=True, parents=True)

    # Build model architecture (needed for load_from_checkpoint)
    startfilter = config.parameters.startfilter
    start_kernel = config.parameters.start_kernel
    mid_kernel = config.parameters.mid_kernel
    depth = config.parameters.depth
    growth_rate = config.parameters.growth_rate
    pool_first = config.parameters.pool_first
    event_position_label = config.parameters.event_position_label

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
        map_location='cpu',
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
        nms_space=nms_space,
        nms_time=nms_time,
        batch_size_predict=batch_size_predict,
    )

    # Get raw tif files — only test_dataset.tif if test_pred mode
    if test_pred:
        raw_files = [os.path.join(raw_timelapses_dir, "test_dataset.tif")]
    else:
        raw_files = sorted(glob(os.path.join(raw_timelapses_dir, config.parameters.file_type)))

    print(f"Found {len(raw_files)} raw timelapse files")
    print(f"Event threshold: {event_threshold}, NMS space: {nms_space}, NMS time: {nms_time}")

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
            for event_name in df['event_name'].unique():
                event_df = df[df['event_name'] == event_name]
                output_df = event_df[['time', 'z', 'y', 'x', 'confidence']].rename(columns={'time': 't'})
                csv_filename = f"{os.path.splitext(raw_basename)[0]}_oneat_{event_name}.csv"
                csv_path = os.path.join(predictions_dir, csv_filename)
                output_df.to_csv(csv_path, index=False)
                print(f"Saved {len(event_df)} {event_name} detections to: {csv_path}")
        else:
            print("No events detected")

    print("\nPrediction complete!")


if __name__ == "__main__":
    main()

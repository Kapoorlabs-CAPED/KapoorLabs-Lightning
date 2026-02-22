from pathlib import Path
import os
import hydra
from hydra.core.config_store import ConfigStore
from scenario_predict_oneat import OneatPredictClass
import torch
from glob import glob
import pandas as pd
from kapoorlabs_lightning.utils import load_checkpoint_model
from kapoorlabs_lightning.oneat_module import OneatActionModule
from kapoorlabs_lightning.oneat_prediction_dataset import OneatPredictionDataset
from kapoorlabs_lightning.nms_utils import nms_space_time, group_detections_by_event
from torch.utils.data import DataLoader
from lightning import Trainer


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

    # NMS parameters
    nms_space = config.parameters.nms_space
    nms_time = config.parameters.nms_time

    # Event parameters
    event_names = config.parameters.event_name

    # Data paths
    base_data_dir = config.experiment_data_paths.base_data_dir
    raw_timelapses_dir = os.path.join(base_data_dir, config.experiment_data_paths.raw_timelapses)
    seg_timelapses_dir = os.path.join(base_data_dir, config.experiment_data_paths.seg_timelapses)
    predictions_dir = os.path.join(base_data_dir, config.experiment_data_paths.oneat_predictions)

    # Model checkpoint path from config
    log_path = config.train_data_paths.log_path
    ckpt_path = load_checkpoint_model(log_path)

    if ckpt_path is None:
        raise ValueError(f"No checkpoint found in {log_path}")

    print(f"Loading model from: {ckpt_path}")

    # Create predictions directory
    Path(predictions_dir).mkdir(exist_ok=True, parents=True)

    # Load model from checkpoint using Lightning
    lightning_model = OneatActionModule.load_from_checkpoint(
        ckpt_path,
        map_location='cpu',
        imagex=imagex,
        imagey=imagey,
        imagez=imagez,
        size_tminus=size_tminus,
        size_tplus=size_tplus,
        event_names=event_names,
        num_classes=num_classes,
    )

    # Get all raw tif files
    raw_files = sorted(glob(os.path.join(raw_timelapses_dir, config.parameters.file_type)))

    print(f"Found {len(raw_files)} raw timelapse files")

    # Create Lightning Trainer for prediction
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
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

        # Create dataloader
        pred_dataloader = DataLoader(
            pred_dataset,
            batch_size=1,  # Process one timepoint at a time
            shuffle=False,
            num_workers=0,  # Set to 0 for prediction
        )

        # Run prediction using Lightning Trainer
        print("Running predictions...")
        predictions = trainer.predict(lightning_model, pred_dataloader)

        # Flatten predictions (each batch returns a list of detections)
        all_detections = []
        for batch_detections in predictions:
            all_detections.extend(batch_detections)

        print(f"Total detections before NMS: {len(all_detections)}")

        # Apply NMS in space and time
        if len(all_detections) > 0:
            # Group by event type first
            grouped_detections = group_detections_by_event(all_detections)

            all_nms_detections = []
            for event_name, event_detections in grouped_detections.items():
                print(f"Applying NMS to {len(event_detections)} {event_name} detections...")
                nms_detections = nms_space_time(event_detections, nms_space=nms_space, nms_time=nms_time)
                print(f"After NMS: {len(nms_detections)} {event_name} detections")
                all_nms_detections.extend(nms_detections)

            # Save predictions to CSV
            if len(all_nms_detections) > 0:
                df = pd.DataFrame(all_nms_detections)

                # Group by event type and save separate CSV files
                for event_name in df['event_name'].unique():
                    event_df = df[df['event_name'] == event_name]
                    # Select only relevant columns for ONEAT CSV format
                    output_df = event_df[['time', 'z', 'y', 'x']].rename(columns={'time': 't'})
                    csv_filename = f"{os.path.splitext(raw_basename)[0]}_oneat_{event_name}.csv"
                    csv_path = os.path.join(predictions_dir, csv_filename)
                    output_df.to_csv(csv_path, index=False)
                    print(f"Saved {len(event_df)} {event_name} detections to: {csv_path}")
            else:
                print("No events detected after NMS")
        else:
            print("No events detected")

    print("\nPrediction complete!")


if __name__ == "__main__":
    main()

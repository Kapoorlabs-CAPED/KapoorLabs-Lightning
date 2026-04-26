#!/usr/bin/env python3
"""
Demo prediction script — runs ONEAT on user-uploaded files.

This script runs on Jean Zay compute nodes via SLURM.
It reads raw + seg TIF from the demo uploads directory,
runs ONEAT prediction, and writes results CSV to the demo results directory.

Usage (called by SLURM script, not directly):
    python demo_predict.py --job-id <job_id> [--checkpoint <path>]
"""

import argparse
import json

from pathlib import Path
from glob import glob

import pandas as pd
import torch
from torch.utils.data import DataLoader
from lightning import Trainer

from kapoorlabs_lightning.oneat_module import OneatActionModule
from kapoorlabs_lightning.oneat_presets import OneatEvalPreset
from kapoorlabs_lightning.oneat_prediction_dataset import OneatPredictionDataset
from kapoorlabs_lightning.pytorch_models import DenseVollNet
from kapoorlabs_lightning.utils import load_checkpoint_model


# Paths on Jean Zay lustre filesystem
LUSTRE_BASE = Path("/lustre/fsn1/projects/rech/jsy/uzj81mi")
DEMO_BASE = LUSTRE_BASE / "demo"
DEFAULT_MODEL_DIR = LUSTRE_BASE / "oneat_mitosis_model_adam_heavy"

# Default model architecture (must match training)
DEFAULT_PARAMS = {
    "imagex": 64,
    "imagey": 64,
    "imagez": 8,
    "size_tminus": 1,
    "size_tplus": 1,
    "startfilter": 64,
    "growth_rate": 32,
    "start_kernel": 7,
    "mid_kernel": 3,
    "depth_0": 12,
    "depth_1": 24,
    "depth_2": 16,
    "pool_first": True,
    "num_classes": 2,
    "nms_space": 10,
    "nms_time": 2,
    "pmin": 1.0,
    "pmax": 99.8,
    "event_threshold": 0.999,
    "event_names": ["normal", "mitosis"],
}


def load_params_from_config(config_path):
    """Load model architecture params from a training_config.json.

    Merges the file's `parameters` block over DEFAULT_PARAMS and flattens
    the nested `depth` dict so downstream keys (`depth_0`, `depth_1`, ...) work.
    """
    with open(config_path) as f:
        cfg = json.load(f)
    params = dict(DEFAULT_PARAMS)
    file_params = cfg.get("parameters", cfg)
    for k, v in file_params.items():
        if k == "depth" and isinstance(v, dict):
            params.update(v)
        else:
            params[k] = v
    return params


def run_prediction(job_id, checkpoint_path=None, config_path=None):
    uploads_dir = DEMO_BASE / "uploads" / job_id
    results_dir = DEMO_BASE / "results" / job_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # Write a status file so the app can track progress
    status_file = results_dir / "status.txt"
    status_file.write_text("running")

    try:
        # Find uploaded files
        raw_files = sorted(glob(str(uploads_dir / "raw_*.tif")))
        seg_files = sorted(glob(str(uploads_dir / "seg_*.tif")))

        if not raw_files or not seg_files:
            status_file.write_text("error: no input files found")
            return

        raw_path = raw_files[0]
        seg_path = seg_files[0]
        print(f"Raw: {raw_path}")
        print(f"Seg: {seg_path}")

      

        # Find checkpoint
        if checkpoint_path is None:
            checkpoint_path = load_checkpoint_model(str(DEFAULT_MODEL_DIR))
        if checkpoint_path is None:
            status_file.write_text("error: no checkpoint found")
            return

        print(f"Checkpoint: {checkpoint_path}")

        if config_path:
            p = load_params_from_config(config_path)
            print(f"Config: {config_path}")
        else:
            p = DEFAULT_PARAMS
        imaget = p["size_tminus"] + p["size_tplus"] + 1
        input_shape = (imaget, p["imagez"], p["imagey"], p["imagex"])
        depth = {"depth_0": p["depth_0"], "depth_1": p["depth_1"], "depth_2": p["depth_2"]}

        # Build model
        network = DenseVollNet(
            input_shape=input_shape,
            categories=p["num_classes"],
            box_vector=8,
            start_kernel=p["start_kernel"],
            mid_kernel=p["mid_kernel"],
            startfilter=p["startfilter"],
            growth_rate=p["growth_rate"],
            depth=depth,
            pool_first=p["pool_first"],
        )

        eval_transforms = OneatEvalPreset(
            percentile_norm=True,
            pmin=p["pmin"],
            pmax=p["pmax"],
        )

        status_file.write_text("loading_model")

        lightning_model = OneatActionModule.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            network=network,
            eval_transforms=eval_transforms,
            imagex=p["imagex"],
            imagey=p["imagey"],
            imagez=p["imagez"],
            size_tminus=p["size_tminus"],
            size_tplus=p["size_tplus"],
            event_names=p["event_names"],
            num_classes=p["num_classes"],
            event_threshold=p["event_threshold"],
            nms_space=p["nms_space"],
            nms_time=p["nms_time"]
        )

        # Dataset
        pred_dataset = OneatPredictionDataset(
            raw_file=raw_path,
            seg_file=seg_path,
            size_tminus=p["size_tminus"],
            size_tplus=p["size_tplus"],
            normalize=True,
            pmin=p["pmin"],
            pmax=p["pmax"],
            chunk_steps=50,
        )

        pred_dataloader = DataLoader(
            pred_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
        )

        status_file.write_text("predicting")

        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )

        predictions = trainer.predict(lightning_model, pred_dataloader)

        all_detections = []
        for batch_detections in predictions:
            all_detections.extend(batch_detections)

        print(f"Total detections before NMS: {len(all_detections)}")

        status_file.write_text("postprocessing")

        if len(all_detections) > 0:
           
                df = pd.DataFrame(all_detections)
                display_df = df[["time", "z", "y", "x", "score",  "event_name", "cell_id"]].copy()
                display_df = display_df.rename(columns={"time": "t"})
                csv_path = results_dir / "oneat_detections.csv"
                display_df.to_csv(csv_path, index=False)
                print(f"Results saved: {csv_path}")
           
        else:
            (results_dir / "oneat_detections.csv").write_text("t,z,y,x,score,event_name,cell_id\n")
            print("No events detected")

        status_file.write_text("done")

    except Exception as e:
        status_file.write_text(f"error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Demo ONEAT prediction")
    parser.add_argument("--job-id", required=True, help="Unique job identifier")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--config", default=None, help="Path to training_config.json with model params")
    args = parser.parse_args()

    run_prediction(args.job_id, checkpoint_path=args.checkpoint, config_path=args.config)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compute classification accuracy on the validation set of ONEAT H5 files.

Loads a trained model checkpoint, runs inference on the val split of each
H5 dataset, and reports per-class + overall accuracy as a pandas DataFrame.
Results are saved to a CSV file.

Usage:
    python compute_val_accuracy.py train_data_paths=gwdg
    python compute_val_accuracy.py train_data_paths=gwdg_spheroids
    python compute_val_accuracy.py train_data_paths=gwdg_combined

    # Run all three and aggregate:
    python compute_val_accuracy.py --multirun train_data_paths=gwdg,gwdg_spheroids,gwdg_combined
"""

import os
import hydra
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader
from scenario_val_accuracy import ValAccuracyConfig
from kapoorlabs_lightning.utils import load_checkpoint_model
from kapoorlabs_lightning.oneat_module import OneatActionModule
from kapoorlabs_lightning.oneat_presets import OneatEvalPreset
from kapoorlabs_lightning.pytorch_datasets import H5VisionDataset
from kapoorlabs_lightning.pytorch_models import DenseVollNet


configstore = ConfigStore.instance()
configstore.store(name="ValAccuracyConfig", node=ValAccuracyConfig)


@hydra.main(
    config_path="../conf", config_name="scenario_val_accuracy"
)
def main(config: ValAccuracyConfig):
    # Model architecture
    imagex = config.parameters.imagex
    imagey = config.parameters.imagey
    imagez = config.parameters.imagez
    size_tminus = config.parameters.size_tminus
    size_tplus = config.parameters.size_tplus
    num_classes = config.parameters.num_classes
    event_names = config.parameters.event_name
    batch_size = config.parameters.batch_size
    num_workers = config.parameters.num_workers
    accelerator = config.parameters.accelerator
    devices = config.parameters.devices

    # DenseVollNet architecture params
    startfilter = config.parameters.startfilter
    start_kernel = config.parameters.start_kernel
    mid_kernel = config.parameters.mid_kernel
    depth = config.parameters.depth
    growth_rate = config.parameters.growth_rate
    pool_first = config.parameters.pool_first
    event_position_label = config.parameters.event_position_label

    # Normalization
    percentile_norm = config.parameters.percentile_norm
    pmin = config.parameters.pmin
    pmax = config.parameters.pmax

    # Paths
    base_data_dir = config.train_data_paths.base_data_dir
    h5_filename = config.train_data_paths.oneat_h5_file
    log_path = config.train_data_paths.log_path
    experiment_name = config.train_data_paths.experiment_name
    h5_file = os.path.join(base_data_dir, h5_filename)

    print(f"\n{'='*60}")
    print(f"Dataset: {h5_filename}")
    print(f"Experiment: {experiment_name}")
    print(f"H5 path: {h5_file}")
    print(f"Log path: {log_path}")
    print(f"{'='*60}")

    # Load checkpoint
    ckpt_path = load_checkpoint_model(log_path)
    if ckpt_path is None:
        raise ValueError(f"No checkpoint found in {log_path}")
    print(f"Checkpoint: {ckpt_path}")

    # Build model architecture (needed for load_from_checkpoint)
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

    # Eval transforms (same as validation during training)
    eval_transforms = OneatEvalPreset(
        percentile_norm=percentile_norm,
        pmin=pmin,
        pmax=pmax,
    )

    # Load model from checkpoint
    device = 'cuda' if accelerator == 'cuda' and torch.cuda.is_available() else 'cpu'
    lightning_model = OneatActionModule.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        network=network,
        loss_func=None,
        optim_func=None,
        imagex=imagex,
        imagey=imagey,
        imagez=imagez,
        size_tminus=size_tminus,
        size_tplus=size_tplus,
        event_names=event_names,
        num_classes=num_classes,
        eval_transforms=eval_transforms,
    )
    lightning_model.eval()
    lightning_model.to(device)

    # Load val dataset
    val_dataset = H5VisionDataset(
        h5_file=h5_file,
        split="val",
        transforms=eval_transforms,
        num_classes=num_classes,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print(f"Val samples: {len(val_dataset)}")

    # Run inference
    all_preds = []
    all_true = []
    box_vector_len = 8

    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = lightning_model(images)

            # Squeeze spatial dims if present
            if outputs.dim() > 2:
                outputs = outputs.squeeze(-1).squeeze(-1).squeeze(-1)

            # Model output: [categories (num_classes), box_vector (8)]
            pred_classes = torch.argmax(outputs[:, :num_classes], dim=1)
            # GT label: [box_vector (8), categories (one-hot)]
            true_classes = torch.argmax(labels[:, box_vector_len:], dim=1)

            all_preds.append(pred_classes.cpu())
            all_true.append(true_classes.cpu())

    all_preds = torch.cat(all_preds)
    all_true = torch.cat(all_true)

    # Compute metrics
    overall_correct = (all_preds == all_true).sum().item()
    overall_accuracy = overall_correct / len(all_true)

    rows = []
    for class_idx, class_name in enumerate(event_names):
        mask = all_true == class_idx
        class_total = mask.sum().item()
        if class_total > 0:
            class_correct = ((all_preds == class_idx) & mask).sum().item()
            class_accuracy = class_correct / class_total
            # Precision: of all predicted as this class, how many are correct
            pred_mask = all_preds == class_idx
            pred_total = pred_mask.sum().item()
            precision = class_correct / pred_total if pred_total > 0 else 0.0
        else:
            class_correct = 0
            class_accuracy = 0.0
            precision = 0.0

        rows.append({
            'dataset': h5_filename,
            'experiment': experiment_name,
            'class_name': class_name,
            'class_idx': class_idx,
            'total_samples': class_total,
            'correct': class_correct,
            'accuracy': round(class_accuracy, 4),
            'precision': round(precision, 4),
        })

    # Overall row
    rows.append({
        'dataset': h5_filename,
        'experiment': experiment_name,
        'class_name': 'OVERALL',
        'class_idx': -1,
        'total_samples': len(all_true),
        'correct': overall_correct,
        'accuracy': round(overall_accuracy, 4),
        'precision': round(overall_accuracy, 4),
    })

    df = pd.DataFrame(rows)

    # Display results
    print(f"\n{'='*60}")
    print(f"Validation Accuracy Results: {h5_filename}")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"{'='*60}\n")

    # Save to CSV
    output_csv = os.path.join(log_path, f"val_accuracy_{experiment_name}.csv")
    df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")

    return df


if __name__ == "__main__":
    main()

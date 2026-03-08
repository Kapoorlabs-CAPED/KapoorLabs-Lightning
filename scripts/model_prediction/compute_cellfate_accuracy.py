#!/usr/bin/env python3
"""
Compute cell fate classification accuracy on the validation set.

Loads a trained model checkpoint, runs inference on the val split of the
H5 file, and reports per-class + overall accuracy as a pandas DataFrame.

Usage:
    python compute_cellfate_accuracy.py
    python compute_cellfate_accuracy.py train_data_paths=cellfate_default
"""

import os
import hydra
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader
from scenario_predict_cellfate import CellFatePredictClass
from kapoorlabs_lightning.utils import load_checkpoint_model
from kapoorlabs_lightning.cellfate_module import CellFateModule
from kapoorlabs_lightning.pytorch_datasets import H5MitosisDataset
from kapoorlabs_lightning.pytorch_models import InceptionNet, DenseNet, MitosisNet


configstore = ConfigStore.instance()
configstore.store(name="CellFatePredictClass", node=CellFatePredictClass)


@hydra.main(
    config_path="../conf", config_name="scenario_predict_cellfate"
)
def main(config: CellFatePredictClass):
    num_classes = config.parameters.num_classes
    batch_size = config.parameters.batch_size
    num_workers = config.parameters.num_workers
    accelerator = config.parameters.accelerator
    model_choice = config.parameters.model_choice

    # Model architecture
    growth_rate = config.parameters.growth_rate
    block_config = tuple(config.parameters.block_config)
    num_init_features = config.parameters.num_init_features
    bottleneck_size = config.parameters.bottleneck_size
    kernel_size = config.parameters.kernel_size
    attn_heads = config.parameters.attn_heads
    seq_len = config.parameters.seq_len

    # Paths
    base_data_dir = config.train_data_paths.base_data_dir
    h5_filename = config.train_data_paths.cellfate_h5_file
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

    # Load val dataset to get input_channels
    val_dataset = H5MitosisDataset(
        h5_file,
        "val_arrays",
        "val_labels",
        num_classes=num_classes,
        transforms=None,
    )
    input_channels = val_dataset.input_channels

    # Build model architecture
    if model_choice == "inception":
        network = InceptionNet(
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
        network = DenseNet(
            input_channels=input_channels,
            num_classes=num_classes,
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=num_init_features,
            bottleneck_size=bottleneck_size,
            kernel_size=kernel_size,
        )
    elif model_choice == "mitosisnet":
        network = MitosisNet(
            input_channels=input_channels,
            num_classes=num_classes,
        )

    # Load model from checkpoint
    device = 'cuda' if accelerator == 'cuda' and torch.cuda.is_available() else 'cpu'
    lightning_model = CellFateModule.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        network=network,
        weights_only=False,
    )
    lightning_model.eval()
    lightning_model.to(device)

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

    with torch.no_grad():
        for batch in val_loader:
            arrays, labels = batch
            arrays = arrays.to(device)
            labels = labels.to(device)

            outputs = lightning_model(arrays)
            pred_classes = torch.argmax(outputs, dim=1)

            all_preds.append(pred_classes.cpu())
            all_true.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_true = torch.cat(all_true)

    # Compute metrics
    overall_correct = (all_preds == all_true).sum().item()
    overall_accuracy = overall_correct / len(all_true)

    # Try to load class names from config json if available
    class_names = [f"class_{i}" for i in range(num_classes)]
    config_json = os.path.join(log_path, experiment_name + ".json")
    if os.path.exists(config_json):
        import json
        with open(config_json) as f:
            saved_config = json.load(f)
            if "class_names" in saved_config:
                class_names = saved_config["class_names"]

    rows = []
    for class_idx in range(num_classes):
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        mask = all_true == class_idx
        class_total = mask.sum().item()
        if class_total > 0:
            class_correct = ((all_preds == class_idx) & mask).sum().item()
            class_accuracy = class_correct / class_total
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

    print(f"\n{'='*60}")
    print(f"Cell Fate Validation Accuracy: {h5_filename}")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"{'='*60}\n")

    output_csv = os.path.join(log_path, f"val_accuracy_{experiment_name}.csv")
    df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")

    return df


if __name__ == "__main__":
    main()

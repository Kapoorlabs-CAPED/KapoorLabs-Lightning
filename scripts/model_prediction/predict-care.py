"""
CARE denoising prediction script.

Applies a trained CARE UNet model to denoise 3D microscopy volumes
using tiled prediction with overlap blending.
"""

import os
from glob import glob
from pathlib import Path

import hydra
import torch
from hydra.core.config_store import ConfigStore
from lightning import Trainer
from tifffile import imread, imwrite
from torch.utils.data import DataLoader

from careamics.models.unet import UNet

from kapoorlabs_lightning.care_dataset import CarePredictionDataset, compute_tile_shape
from kapoorlabs_lightning.care_module import CareModule, stitch_tiles
from kapoorlabs_lightning.oneat_transforms import PercentileNormalize
from kapoorlabs_lightning.utils import load_checkpoint_model

from scenario_predict_care import CarePredictClass

configstore = ConfigStore.instance()
configstore.store(name="CarePredictClass", node=CarePredictClass)


@hydra.main(config_path="../conf", config_name="scenario_predict_care", version_base='1.3')
def main(config: CarePredictClass):
    # Model architecture
    unet_depth = config.parameters.unet_depth
    num_channels_init = config.parameters.num_channels_init
    use_batch_norm = config.parameters.use_batch_norm

    # Prediction parameters
    devices = config.parameters.devices
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    n_tiles = config.parameters.n_tiles
    tile_overlap = config.parameters.tile_overlap
    batch_size = config.parameters.batch_size
    pmin = config.parameters.pmin
    pmax = config.parameters.pmax
    file_type = config.parameters.file_type

    # Data paths
    base_data_dir = config.experiment_data_paths.base_data_dir
    input_dir = os.path.join(base_data_dir, config.experiment_data_paths.input_dir)
    output_dir = os.path.join(base_data_dir, config.experiment_data_paths.output_dir)

    # Model checkpoint
    log_path = config.train_data_paths.log_path
    ckpt_path = load_checkpoint_model(log_path)

    if ckpt_path is None:
        raise ValueError(f"No checkpoint found in {log_path}")

    print(f"Loading model from: {ckpt_path}")

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Build UNet
    network = UNet(
        conv_dims=3,
        in_channels=1,
        num_classes=1,
        depth=unet_depth,
        num_channels_init=num_channels_init,
        use_batch_norm=use_batch_norm,
    )

    # Normalizer for prediction tiles
    normalizer = PercentileNormalize(pmin=pmin, pmax=pmax)

    # Load model
    model = CareModule.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
        network=network,
        n_tiles=n_tiles,
        tile_overlap=tile_overlap,
    )
    model.eval()

    # Find input files
    input_files = sorted(glob(os.path.join(input_dir, file_type)))
    print(f"Found {len(input_files)} input files")
    print(f"Tile config: n_tiles={n_tiles}, overlap={tile_overlap}")

    # Create trainer for prediction
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    for input_file in input_files:
        basename = os.path.basename(input_file)
        print(f"\nProcessing: {basename}")

        volume = imread(input_file)

        # Handle multi-channel: take first channel
        if volume.ndim == 4:
            volume = volume[:, 0] if volume.shape[1] < volume.shape[0] else volume[0]

        print(f"  Volume shape: {volume.shape}")

        # Compute tile shape from n_tiles
        tile_shape = compute_tile_shape(volume.shape, n_tiles)
        print(f"  Tile shape: {tile_shape}")

        # Create prediction dataset
        pred_dataset = CarePredictionDataset(
            volume=volume,
            tile_shape=tile_shape,
            overlap=tile_overlap,
            normalizer=normalizer,
        )

        pred_dataloader = DataLoader(
            pred_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        print(f"  {len(pred_dataset)} tiles, running prediction...")

        # Run tiled prediction
        predictions = trainer.predict(model, pred_dataloader)

        # Stitch tiles back together
        denoised = stitch_tiles(predictions, volume.shape, tile_overlap)

        # Save output
        output_path = os.path.join(output_dir, basename)
        imwrite(output_path, denoised)
        print(f"  Saved: {output_path}")

    print("\nPrediction complete!")


if __name__ == "__main__":
    main()

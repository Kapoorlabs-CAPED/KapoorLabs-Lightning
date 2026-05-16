"""
ROI segmentation prediction script.

The ROI Mask-UNet is **inherently 2D** (see
``conf/parameters/roi.yaml`` → ``conv_dims: 2`` and the original
``VollSeg.utils.VollSeg_unet`` flow). This script handles both:

- **2D inputs (YX)** — direct tiled prediction on the image.
- **3D inputs (ZYX)** — max-Z projection first, then the same 2D
  prediction; the resulting 2D mask is broadcast back to ZYX so the
  output has the same shape as the input.

Architecture knobs are loaded from ``training_config.json`` next to
the checkpoint when present (the canonical Hydra dump), with the
Hydra ``parameters/roi.yaml`` as the fallback. This means a model
trained with a different ``conv_dims`` / ``num_channels_init`` /
``unet_depth`` than the current Hydra config can still be used for
prediction without editing the yaml.

After prediction we apply ``skimage.filters.threshold_multiotsu``
(2 classes) to binarise the model response, label connected
components, drop small ones, and fill holes — same post-processing
as the original ``VollSeg2D`` / ``VollSeg_unet``.
"""

import json
import os
from glob import glob
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from lightning import Trainer
from skimage.filters import threshold_multiotsu
from skimage.measure import label
from skimage.morphology import remove_small_objects
from tifffile import imread, imwrite
from torch.utils.data import DataLoader

from careamics.models.unet import UNet

from kapoorlabs_lightning.care_dataset import CarePredictionDataset, compute_tile_shape
from kapoorlabs_lightning.care_module import CareModule, stitch_tiles
from kapoorlabs_lightning.oneat_transforms import PercentileNormalize
from kapoorlabs_lightning.utils import load_checkpoint_model

from scenario_predict_roi import CarePredictClass
from _arch_loader import load_arch_from_training_config


configstore = ConfigStore.instance()
configstore.store(name="CarePredictClass", node=CarePredictClass)

# Default minimum ROI region size (voxels) — same threshold the
# original VollSeg_unet pipeline uses. Bump via the env var if your
# data needs a different floor.
DEFAULT_MIN_SIZE_MASK = int(os.environ.get("ROI_MIN_SIZE_MASK", "100"))


@hydra.main(config_path="../conf", config_name="scenario_predict_roi", version_base="1.3")
def main(config: CarePredictClass):
    # Defaults from Hydra; the JSON next to the checkpoint wins.
    cfg_params = config.parameters

    # Paths
    base_data_dir = config.experiment_data_paths.base_data_dir
    input_dir = os.path.join(base_data_dir, config.experiment_data_paths.input_dir)
    output_dir = os.path.join(base_data_dir, config.experiment_data_paths.output_dir)
    log_path = config.train_data_paths.log_path

    ckpt_path = load_checkpoint_model(log_path)
    if ckpt_path is None:
        raise ValueError(f"No checkpoint found in {log_path}")
    print(f"Loading model from: {ckpt_path}")

    json_params = load_arch_from_training_config(log_path)
    if json_params:
        print(f"Loaded arch from {log_path}/training_config.json")
    else:
        print(f"No training_config.json in {log_path} — falling back to parameters/roi.yaml")

    conv_dims = json_params.get("conv_dims", cfg_params.conv_dims)
    unet_depth = json_params.get("unet_depth", cfg_params.unet_depth)
    num_channels_init = json_params.get("num_channels_init", cfg_params.num_channels_init)
    use_batch_norm = json_params.get("use_batch_norm", cfg_params.use_batch_norm)
    in_channels = json_params.get("in_channels", 1)
    num_classes = json_params.get("num_classes", 1)

    if conv_dims != 2:
        print(
            f"  ⚠ conv_dims={conv_dims} — the ROI script is designed around a 2D "
            f"model. Predictions will run anyway but the MIP step will not apply."
        )

    # Inference knobs
    devices = cfg_params.devices
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    n_tiles = list(cfg_params.n_tiles)
    tile_overlap = cfg_params.tile_overlap
    batch_size = cfg_params.batch_size
    pmin = cfg_params.pmin
    pmax = cfg_params.pmax
    file_type = cfg_params.file_type

    Path(output_dir).mkdir(exist_ok=True, parents=True)

    network = UNet(
        conv_dims=conv_dims,
        in_channels=in_channels,
        num_classes=num_classes,
        depth=unet_depth,
        num_channels_init=num_channels_init,
        use_batch_norm=use_batch_norm,
    )
    normalizer = PercentileNormalize(pmin=pmin, pmax=pmax)

    model = CareModule.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
        network=network,
        n_tiles=n_tiles,
        tile_overlap=tile_overlap,
    )
    model.eval()

    input_files = sorted(glob(os.path.join(input_dir, file_type)))
    print(f"Found {len(input_files)} input files")
    print(f"Tile config: n_tiles={n_tiles}, overlap={tile_overlap}")

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

        # Drop channel axis if present (single-channel ROI model).
        if volume.ndim == 4:
            volume = volume[:, 0] if volume.shape[1] < volume.shape[0] else volume[0]
        original_shape = volume.shape
        print(f"  Input shape: {original_shape}")

        # 2D-on-3D: project along Z so the 2D model sees one 2D image.
        # This matches `VollSeg_unet`: a single ROI mask gates the whole stack.
        if volume.ndim == 3 and conv_dims == 2:
            image_2d = np.amax(volume, axis=0)
            print(f"  MIP along Z → 2D shape: {image_2d.shape}")
        else:
            image_2d = volume

        # Tiled prediction
        tile_shape = compute_tile_shape(image_2d.shape, n_tiles)
        pred_dataset = CarePredictionDataset(
            volume=image_2d.astype(np.float32),
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
        predictions = trainer.predict(model, pred_dataloader)
        roi_response = stitch_tiles(predictions, image_2d.shape, tile_overlap)

        # Multi-Otsu (2 classes) → binarise — same as original VollSeg2D.
        try:
            thresholds = threshold_multiotsu(roi_response, classes=2)
            regions = np.digitize(roi_response, bins=thresholds)
        except ValueError:
            regions = roi_response

        roi_mask = label(regions > 0)
        if DEFAULT_MIN_SIZE_MASK > 0:
            roi_mask = remove_small_objects(
                roi_mask.astype(np.int32), min_size=DEFAULT_MIN_SIZE_MASK,
            )
            roi_mask = label(roi_mask > 0)             # re-label after removal

        # Broadcast the 2D mask back to ZYX when the input was 3D.
        if len(original_shape) == 3 and conv_dims == 2:
            roi_mask = np.broadcast_to(roi_mask, original_shape).copy()

        roi_mask = roi_mask.astype(np.uint16)
        output_path = os.path.join(output_dir, basename)
        imwrite(output_path, roi_mask)
        print(
            f"  Saved: {output_path}   "
            f"({int(roi_mask.max())} ROI regions, shape={roi_mask.shape})"
        )

    print("\nPrediction complete!")


if __name__ == "__main__":
    main()

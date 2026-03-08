"""
ONEAT Event Detection - Gradio App for Hugging Face Spaces

Upload raw + segmentation timelapse TIF files, select a model checkpoint,
and run ONEAT prediction to detect spatio-temporal events (e.g. mitosis).
Results are displayed as a table and downloadable CSV.
"""

import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
from tifffile import imread

from kapoorlabs_lightning.oneat_module import OneatActionModule
from kapoorlabs_lightning.oneat_presets import OneatEvalPreset
from kapoorlabs_lightning.oneat_prediction_dataset import OneatPredictionDataset
from kapoorlabs_lightning.nms_utils import nms_space_time, group_detections_by_event
from kapoorlabs_lightning.pytorch_models import DenseVollNet
from torch.utils.data import DataLoader
from lightning import Trainer

# Directory where .ckpt files are stored (uploaded with the space)
MODELS_DIR = Path(__file__).parent / "models"


def get_available_models():
    """List all .ckpt files in the models directory."""
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return []
    ckpts = sorted(MODELS_DIR.glob("*.ckpt"))
    return [f.name for f in ckpts]


def run_prediction(
    raw_file,
    seg_file,
    model_name,
    num_classes,
    imagex,
    imagey,
    imagez,
    size_tminus,
    size_tplus,
    startfilter,
    start_kernel,
    mid_kernel,
    growth_rate,
    depth_0,
    depth_1,
    depth_2,
    pool_first,
    nms_space,
    nms_time,
    event_names_str,
    pmin,
    pmax,
    normalize,
    progress=gr.Progress(),
):
    """Run ONEAT prediction on uploaded timelapse files."""

    if raw_file is None or seg_file is None:
        return None, None, None, "Please upload both raw and segmentation timelapse files."

    if not model_name:
        return None, None, None, "Please select a model checkpoint."

    ckpt_path = MODELS_DIR / model_name

    if not ckpt_path.exists():
        return None, None, None, f"Model checkpoint not found: {ckpt_path}"

    event_names = [e.strip() for e in event_names_str.split(",")]

    progress(0.1, desc="Building model architecture...")

    # Build the network architecture
    imaget = size_tminus + size_tplus + 1
    input_shape = (imaget, imagez, imagey, imagex)
    depth = {"depth_0": depth_0, "depth_1": depth_1, "depth_2": depth_2}
    box_vector = 8  # ONEAT standard: x, y, z, t, h, w, d, c

    network = DenseVollNet(
        input_shape=input_shape,
        categories=num_classes,
        box_vector=box_vector,
        start_kernel=start_kernel,
        mid_kernel=mid_kernel,
        startfilter=startfilter,
        growth_rate=growth_rate,
        depth=depth,
        pool_first=pool_first,
    )

    # Create eval transforms
    eval_transforms = OneatEvalPreset(
        percentile_norm=True,
        pmin=pmin,
        pmax=pmax,
    )

    progress(0.2, desc="Loading model checkpoint...")

    # Load model from checkpoint
    lightning_model = OneatActionModule.load_from_checkpoint(
        str(ckpt_path),
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
    )

    progress(0.3, desc="Loading timelapse data...")

    # Create prediction dataset
    pred_dataset = OneatPredictionDataset(
        raw_file=raw_file,
        seg_file=seg_file,
        size_tminus=size_tminus,
        size_tplus=size_tplus,
        normalize=normalize,
        pmin=pmin,
        pmax=pmax,
        chunk_steps=50,
    )

    pred_dataloader = DataLoader(
        pred_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    progress(0.4, desc="Running predictions...")

    # Run prediction
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    predictions = trainer.predict(lightning_model, pred_dataloader)

    # Flatten predictions
    all_detections = []
    for batch_detections in predictions:
        all_detections.extend(batch_detections)

    progress(0.8, desc="Applying NMS...")

    status_msg = f"Total detections before NMS: {len(all_detections)}\n"

    # Apply NMS
    if len(all_detections) > 0:
        grouped_detections = group_detections_by_event(all_detections)

        all_nms_detections = []
        for event_name, event_detections in grouped_detections.items():
            nms_detections = nms_space_time(
                event_detections, nms_space=nms_space, nms_time=nms_time
            )
            status_msg += f"{event_name}: {len(event_detections)} -> {len(nms_detections)} after NMS\n"
            all_nms_detections.extend(nms_detections)

        if len(all_nms_detections) > 0:
            df = pd.DataFrame(all_nms_detections)
            display_df = df[["time", "z", "y", "x", "event_name", "cell_id"]].copy()
            display_df = display_df.rename(columns={"time": "t"})

            # Save CSV to temp file
            csv_path = tempfile.mktemp(suffix=".csv")
            display_df.to_csv(csv_path, index=False)

            status_msg += f"\nTotal detections after NMS: {len(all_nms_detections)}"

            progress(0.9, desc="Preparing visualization...")

            # Create a max-projection viewer image for the middle timepoint
            viewer_image = create_detection_overlay(
                pred_dataset.raw_image, all_nms_detections
            )

            return display_df, csv_path, viewer_image, status_msg
        else:
            status_msg += "\nNo events detected after NMS."
            return pd.DataFrame(), None, None, status_msg
    else:
        status_msg += "\nNo events detected."
        return pd.DataFrame(), None, None, status_msg


def create_detection_overlay(raw_image, detections):
    """
    Create a max-Z-projection with detection markers overlaid.
    Returns a numpy array suitable for Gradio Image display.

    raw_image: (T, Z, Y, X) numpy array
    detections: list of dicts with t, y, x keys
    """
    if raw_image is None or len(detections) == 0:
        return None

    # Group detections by timepoint
    det_by_time = {}
    for d in detections:
        t = d["time"]
        if t not in det_by_time:
            det_by_time[t] = []
        det_by_time[t].append(d)

    # Pick the timepoint with the most detections for display
    best_t = max(det_by_time, key=lambda t: len(det_by_time[t]))
    dets_at_t = det_by_time[best_t]

    # Max Z projection at that timepoint
    frame = raw_image[best_t]  # (Z, Y, X)
    max_proj = np.max(frame, axis=0)  # (Y, X)

    # Normalize to 0-255
    vmin, vmax = np.percentile(max_proj, (1, 99.8))
    if vmax > vmin:
        img = np.clip((max_proj - vmin) / (vmax - vmin), 0, 1)
    else:
        img = np.zeros_like(max_proj, dtype=np.float32)

    # Convert to RGB
    img_rgb = np.stack([img, img, img], axis=-1)
    img_rgb = (img_rgb * 255).astype(np.uint8)

    # Draw detection markers (red circles)
    for d in dets_at_t:
        y, x = int(d["y"]), int(d["x"])
        radius = 5
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy * dy + dx * dx <= radius * radius:
                    py, px = y + dy, x + dx
                    if 0 <= py < img_rgb.shape[0] and 0 <= px < img_rgb.shape[1]:
                        # Red for detections
                        img_rgb[py, px] = [255, 50, 50]

    return img_rgb


# Build Gradio interface
with gr.Blocks(title="ONEAT Event Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # ONEAT Event Detection
        **Spatio-temporal event detection in 3D+T microscopy data**

        Upload your raw timelapse and segmentation timelapse (TIF format),
        select a model, and run predictions. Results include event locations
        (t, z, y, x) with NMS filtering.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input Files")
            raw_input = gr.File(label="Raw Timelapse (TIF)", file_types=[".tif", ".tiff"])
            seg_input = gr.File(label="Segmentation Timelapse (TIF)", file_types=[".tif", ".tiff"])

            gr.Markdown("### Model Selection")
            model_dropdown = gr.Dropdown(
                choices=get_available_models(),
                label="Model Checkpoint",
                info="Place .ckpt files in the 'models/' directory",
            )
            refresh_btn = gr.Button("Refresh Models", size="sm")

            gr.Markdown("### Event Configuration")
            event_names_input = gr.Textbox(
                value="normal, mitosis",
                label="Event Names (comma-separated)",
                info="First event (index 0) is treated as 'normal' and not reported",
            )
            num_classes_input = gr.Number(value=2, label="Number of Classes", precision=0)

        with gr.Column(scale=1):
            gr.Markdown("### Model Architecture")
            with gr.Row():
                imagex_input = gr.Number(value=64, label="Image X", precision=0)
                imagey_input = gr.Number(value=64, label="Image Y", precision=0)
                imagez_input = gr.Number(value=8, label="Image Z", precision=0)
            with gr.Row():
                size_tminus_input = gr.Number(value=1, label="T-minus", precision=0)
                size_tplus_input = gr.Number(value=1, label="T-plus", precision=0)
            with gr.Row():
                startfilter_input = gr.Number(value=64, label="Start Filter", precision=0)
                growth_rate_input = gr.Number(value=32, label="Growth Rate", precision=0)
            with gr.Row():
                start_kernel_input = gr.Number(value=7, label="Start Kernel", precision=0)
                mid_kernel_input = gr.Number(value=3, label="Mid Kernel", precision=0)
            with gr.Row():
                depth0_input = gr.Number(value=12, label="Depth 0", precision=0)
                depth1_input = gr.Number(value=24, label="Depth 1", precision=0)
                depth2_input = gr.Number(value=16, label="Depth 2", precision=0)
            pool_first_input = gr.Checkbox(value=True, label="Pool First")

            gr.Markdown("### NMS & Normalization")
            with gr.Row():
                nms_space_input = gr.Number(value=10, label="NMS Space", precision=0)
                nms_time_input = gr.Number(value=2, label="NMS Time", precision=0)
            with gr.Row():
                pmin_input = gr.Number(value=1.0, label="Percentile Min")
                pmax_input = gr.Number(value=99.8, label="Percentile Max")
            normalize_input = gr.Checkbox(value=True, label="Normalize Timelapse")

    run_btn = gr.Button("Run Prediction", variant="primary", size="lg")

    gr.Markdown("---")
    gr.Markdown("### Results")

    with gr.Row():
        with gr.Column(scale=2):
            results_table = gr.Dataframe(
                label="Detected Events",
                headers=["t", "z", "y", "x", "event_name", "cell_id"],
            )
            csv_output = gr.File(label="Download CSV")
        with gr.Column(scale=1):
            viewer_image = gr.Image(
                label="Detection Overlay (max-Z projection at peak timepoint)",
            )

    status_output = gr.Textbox(label="Status", lines=5)

    # Refresh models list
    def refresh_models():
        models = get_available_models()
        return gr.Dropdown(choices=models)

    refresh_btn.click(fn=refresh_models, outputs=model_dropdown)

    # Run prediction
    run_btn.click(
        fn=run_prediction,
        inputs=[
            raw_input,
            seg_input,
            model_dropdown,
            num_classes_input,
            imagex_input,
            imagey_input,
            imagez_input,
            size_tminus_input,
            size_tplus_input,
            startfilter_input,
            start_kernel_input,
            mid_kernel_input,
            growth_rate_input,
            depth0_input,
            depth1_input,
            depth2_input,
            pool_first_input,
            nms_space_input,
            nms_time_input,
            event_names_input,
            pmin_input,
            pmax_input,
            normalize_input,
        ],
        outputs=[results_table, csv_output, viewer_image, status_output],
    )


if __name__ == "__main__":
    demo.launch()

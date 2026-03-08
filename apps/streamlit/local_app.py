"""
ONEAT Event Detection - Local Streamlit App

Upload raw + segmentation timelapse TIF files, select a model checkpoint,
and run ONEAT prediction locally. Results displayed as table + overlay viewer.

Usage:
    streamlit run local_app.py
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
from tifffile import imread

from kapoorlabs_lightning.oneat_module import OneatActionModule
from kapoorlabs_lightning.oneat_presets import OneatEvalPreset
from kapoorlabs_lightning.oneat_prediction_dataset import OneatPredictionDataset
from kapoorlabs_lightning.nms_utils import nms_space_time, group_detections_by_event
from kapoorlabs_lightning.pytorch_models import DenseVollNet
from torch.utils.data import DataLoader
from lightning import Trainer

MODELS_DIR = Path(__file__).parent / "models"


def get_available_models():
    """List all .ckpt files in the models directory."""
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return []
    return sorted([f.name for f in MODELS_DIR.glob("*.ckpt")])


def create_detection_overlay(raw_image, detections, timepoint):
    """Max-Z-projection with detection markers at a given timepoint."""
    dets_at_t = [d for d in detections if d["time"] == timepoint]

    frame = raw_image[timepoint]  # (Z, Y, X)
    max_proj = np.max(frame, axis=0)  # (Y, X)

    vmin, vmax = np.percentile(max_proj, (1, 99.8))
    if vmax > vmin:
        img = np.clip((max_proj - vmin) / (vmax - vmin), 0, 1)
    else:
        img = np.zeros_like(max_proj, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap="gray")

    if dets_at_t:
        ys = [d["y"] for d in dets_at_t]
        xs = [d["x"] for d in dets_at_t]
        ax.scatter(
            xs, ys, c="red", s=80, marker="o",
            facecolors="none", linewidths=2,
            label=f"Detections ({len(dets_at_t)})",
        )
        ax.legend(loc="upper right", fontsize=9)

    ax.set_title(
        f"Max-Z Projection   t={timepoint}   ({len(dets_at_t)} detections)",
        fontsize=11,
    )
    ax.axis("off")
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="ONEAT Event Detection", layout="wide")

    st.title("ONEAT Event Detection")
    st.markdown("Spatio-temporal event detection in 3D+T microscopy data")

    # Sidebar: all configuration
    st.sidebar.header("Input Files")
    raw_file = st.sidebar.file_uploader(
        "Raw Timelapse (TIF)", type=["tif", "tiff"], key="raw"
    )
    seg_file = st.sidebar.file_uploader(
        "Segmentation Timelapse (TIF)", type=["tif", "tiff"], key="seg"
    )

    st.sidebar.header("Model")
    models = get_available_models()
    if not models:
        st.sidebar.warning("No .ckpt files in models/ directory")
    model_name = st.sidebar.selectbox(
        "Checkpoint", models if models else ["(none)"]
    )

    st.sidebar.header("Event Configuration")
    event_names_str = st.sidebar.text_input(
        "Event Names (comma-separated)", value="normal, mitosis"
    )
    num_classes = st.sidebar.number_input(
        "Number of Classes", value=2, min_value=1, step=1
    )

    st.sidebar.header("Model Architecture")
    col_a, col_b, col_c = st.sidebar.columns(3)
    imagex = col_a.number_input("Image X", value=64, step=1)
    imagey = col_b.number_input("Image Y", value=64, step=1)
    imagez = col_c.number_input("Image Z", value=8, step=1)

    col_d, col_e = st.sidebar.columns(2)
    size_tminus = col_d.number_input("T-minus", value=1, min_value=0, step=1)
    size_tplus = col_e.number_input("T-plus", value=1, min_value=0, step=1)

    col_f, col_g = st.sidebar.columns(2)
    startfilter = col_f.number_input("Start Filter", value=64, step=1)
    growth_rate = col_g.number_input("Growth Rate", value=32, step=1)

    col_h, col_i = st.sidebar.columns(2)
    start_kernel = col_h.number_input("Start Kernel", value=7, step=1)
    mid_kernel = col_i.number_input("Mid Kernel", value=3, step=1)

    col_j, col_k, col_l = st.sidebar.columns(3)
    depth_0 = col_j.number_input("Depth 0", value=12, step=1)
    depth_1 = col_k.number_input("Depth 1", value=24, step=1)
    depth_2 = col_l.number_input("Depth 2", value=16, step=1)

    pool_first = st.sidebar.checkbox("Pool First", value=True)

    st.sidebar.header("NMS & Normalization")
    col_m, col_n = st.sidebar.columns(2)
    nms_space = col_m.number_input("NMS Space", value=10, step=1)
    nms_time = col_n.number_input("NMS Time", value=2, step=1)

    col_o, col_p = st.sidebar.columns(2)
    pmin = col_o.number_input("Percentile Min", value=1.0, step=0.1)
    pmax = col_p.number_input("Percentile Max", value=99.8, step=0.1)
    normalize = st.sidebar.checkbox("Normalize Timelapse", value=True)

    use_gpu = st.sidebar.checkbox(
        "Use GPU (if available)", value=torch.cuda.is_available()
    )

    # Main area
    run_btn = st.button(
        "Run Prediction", type="primary", use_container_width=True
    )

    if run_btn:
        if raw_file is None or seg_file is None:
            st.error("Please upload both raw and segmentation timelapse files.")
            return
        if not models or model_name == "(none)":
            st.error(
                "No model checkpoint available. "
                "Place .ckpt files in the models/ directory."
            )
            return

        ckpt_path = MODELS_DIR / model_name
        event_names = [e.strip() for e in event_names_str.split(",")]
        imaget = size_tminus + size_tplus + 1
        input_shape = (imaget, imagez, imagey, imagex)
        depth = {
            "depth_0": depth_0,
            "depth_1": depth_1,
            "depth_2": depth_2,
        }
        box_vector = 8

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_raw:
            tmp_raw.write(raw_file.read())
            raw_path = tmp_raw.name

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_seg:
            tmp_seg.write(seg_file.read())
            seg_path = tmp_seg.name

        try:
            progress = st.progress(0, text="Building model architecture...")

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

            eval_transforms = OneatEvalPreset(
                percentile_norm=True,
                pmin=pmin,
                pmax=pmax,
            )

            progress.progress(15, text="Loading model checkpoint...")

            accelerator = (
                "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            )

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

            progress.progress(25, text="Loading timelapse data...")

            pred_dataset = OneatPredictionDataset(
                raw_file=raw_path,
                seg_file=seg_path,
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

            progress.progress(35, text="Running predictions...")

            trainer = Trainer(
                accelerator=accelerator,
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=True,
            )

            predictions = trainer.predict(lightning_model, pred_dataloader)

            all_detections = []
            for batch_detections in predictions:
                all_detections.extend(batch_detections)

            progress.progress(75, text="Applying NMS...")

            st.info(f"Total detections before NMS: {len(all_detections)}")

            if len(all_detections) > 0:
                grouped_detections = group_detections_by_event(all_detections)

                all_nms_detections = []
                for event_name_key, event_detections in grouped_detections.items():
                    nms_detections = nms_space_time(
                        event_detections,
                        nms_space=nms_space,
                        nms_time=nms_time,
                    )
                    st.write(
                        f"**{event_name_key}**: {len(event_detections)} "
                        f"-> {len(nms_detections)} after NMS"
                    )
                    all_nms_detections.extend(nms_detections)

                progress.progress(90, text="Preparing results...")

                if len(all_nms_detections) > 0:
                    df = pd.DataFrame(all_nms_detections)
                    display_df = df[
                        ["time", "z", "y", "x", "event_name", "cell_id"]
                    ].copy()
                    display_df = display_df.rename(columns={"time": "t"})

                    st.session_state["detections"] = all_nms_detections
                    st.session_state["display_df"] = display_df
                    st.session_state["raw_image"] = pred_dataset.raw_image
                    st.session_state["num_timepoints"] = (
                        pred_dataset.num_timepoints
                    )

                    csv_path = tempfile.mktemp(suffix=".csv")
                    display_df.to_csv(csv_path, index=False)
                    st.session_state["csv_path"] = csv_path

                    progress.progress(100, text="Done!")
                    st.success(
                        f"Detected {len(all_nms_detections)} events after NMS"
                    )
                else:
                    progress.progress(100, text="Done!")
                    st.warning("No events detected after NMS.")
            else:
                progress.progress(100, text="Done!")
                st.warning("No events detected.")

        finally:
            os.unlink(raw_path)
            os.unlink(seg_path)

    # Results display (persists via session state)
    if (
        "display_df" in st.session_state
        and not st.session_state["display_df"].empty
    ):
        st.markdown("---")

        tab_table, tab_viewer = st.tabs(["Results Table", "Detection Viewer"])

        with tab_table:
            st.subheader(
                f"Detected Events ({len(st.session_state['display_df'])})"
            )
            st.dataframe(
                st.session_state["display_df"],
                use_container_width=True,
                hide_index=True,
            )

            with open(st.session_state["csv_path"], "rb") as f:
                st.download_button(
                    "Download CSV",
                    data=f,
                    file_name="oneat_detections.csv",
                    mime="text/csv",
                )

        with tab_viewer:
            st.subheader("Detection Overlay Viewer")
            detections = st.session_state["detections"]
            raw_image = st.session_state["raw_image"]
            num_timepoints = st.session_state["num_timepoints"]

            det_timepoints = sorted(set(d["time"] for d in detections))

            if det_timepoints:
                selected_t = st.slider(
                    "Timepoint",
                    min_value=0,
                    max_value=num_timepoints - 1,
                    value=det_timepoints[0],
                    step=1,
                )

                n_dets_at_t = sum(
                    1 for d in detections if d["time"] == selected_t
                )
                st.caption(
                    f"Timepoint {selected_t} — {n_dets_at_t} detections"
                    f"  |  Timepoints with events: {det_timepoints}"
                )

                fig = create_detection_overlay(
                    raw_image, detections, selected_t
                )
                st.pyplot(fig)
                plt.close(fig)

                dets_at_t = [
                    d for d in detections if d["time"] == selected_t
                ]
                if dets_at_t:
                    st.dataframe(
                        pd.DataFrame(dets_at_t)[
                            ["time", "z", "y", "x", "event_name", "cell_id"]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.info("No detections to display.")


if __name__ == "__main__":
    main()

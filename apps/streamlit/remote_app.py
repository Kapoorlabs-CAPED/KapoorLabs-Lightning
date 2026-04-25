"""
ONEAT Event Detection - Streamlit App

Upload raw + segmentation timelapse TIF files and run ONEAT prediction
on KapoorLabsHPC via SSH + SLURM. Results appear via shared mount.

Usage:
    streamlit run remote_app.py
"""

import json
import logging
import os
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("oneat-demo")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from tifffile import TiffFile

# kapoorlabslustre mounted locally via sshfs
JEANZAY_MOUNT = Path("/home/debian/jean-zay/demo")
UPLOADS_DIR = JEANZAY_MOUNT / "uploads"
RESULTS_DIR = JEANZAY_MOUNT / "results"
MODELS_DIR = JEANZAY_MOUNT / "models"

# Lustre-side path that Jean Zay compute nodes see
LUSTRE_DEMO = Path("/lustre/fsn1/projects/rech/jsy/uzj81mi/demo")

DEFAULT_RAW = UPLOADS_DIR / "raw_demo_default.tif"
DEFAULT_SEG = UPLOADS_DIR / "seg_demo_default.tif"

# Submit script on the lustre mount
SUBMIT_SCRIPT = JEANZAY_MOUNT / "submit_job.sh"


def discover_models():
    """Scan MODELS_DIR for model directories containing .ckpt + config .json."""
    models = {}
    if not MODELS_DIR.exists():
        return models
    for entry in sorted(MODELS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        ckpts = list(entry.glob("*.ckpt"))
        if not ckpts:
            continue
        best = next((c for c in ckpts if "best" in c.name), ckpts[0])
        config_json = None
        for jf in entry.glob("*.json"):
            if jf.name == "training_config.json":
                continue
            config_json = jf
            break
        training_json = entry / "training_config.json"
        models[entry.name] = {
            "dir": str(entry),
            "ckpt": str(best),
            "config_json": str(config_json) if config_json else None,
            "training_json": str(training_json) if training_json.exists() else None,
        }
    return models


def load_json(path):
    """Read a JSON file, return dict or None."""
    if path is None or not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def read_single_slice(tif_path, t=0, z_mid=None, crop=None):
    """Read a single Z-slice from a TZYX timelapse via page-level access.

    Args:
        crop: Optional dict with x_size, y_size keys. Crop is centered.
    """
    with TiffFile(str(tif_path)) as tif:
        series = tif.series[0]
        shape = series.shape
        n_pages = len(tif.pages)

        if len(shape) >= 4:
            nz = shape[1]
            if z_mid is None:
                z_mid = nz // 2
            page_idx = t * nz + z_mid
        elif len(shape) == 3:
            if z_mid is None:
                z_mid = shape[0] // 2
            page_idx = z_mid
        else:
            page_idx = 0

        page_idx = min(page_idx, n_pages - 1)
        img = np.asarray(tif.pages[page_idx].asarray())

    if crop:
        h, w = img.shape[:2]
        xw = min(crop.get("x_size", w), w)
        yw = min(crop.get("y_size", h), h)
        x0 = max(w // 2 - xw // 2, 0)
        y0 = max(h // 2 - yw // 2, 0)
        img = img[y0:y0+yw, x0:x0+xw]

    return img, shape


def _normalize_slice(img_2d):
    """Percentile-normalize a 2D array to uint8 for display."""
    vmin, vmax = np.percentile(img_2d, (1, 99.8))
    if vmax > vmin:
        normed = np.clip((img_2d.astype(float) - vmin) / (vmax - vmin), 0, 1)
    else:
        normed = np.zeros_like(img_2d, dtype=float)
    return (normed * 255).astype(np.uint8)


def preview_slice_figure(img_2d, title=""):
    """Interactive plotly figure with zoom/pan from a 2D array."""
    img8 = _normalize_slice(img_2d)
    fig = go.Figure(go.Image(z=np.stack([img8, img8, img8], axis=-1)))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
        dragmode="zoom",
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


def create_detection_overlay(raw_slice_2d, detections_df, timepoint):
    """Interactive plotly figure with detections as markers."""
    dets_at_t = detections_df[detections_df["t"] == timepoint]
    img8 = _normalize_slice(raw_slice_2d)
    h, w = img8.shape

    fig = go.Figure(go.Image(z=np.stack([img8, img8, img8], axis=-1)))

    if len(dets_at_t) > 0:
        hover_text = [
            f"x={row['x']:.0f}, y={row['y']:.0f}"
            + (f", z={row['z']:.0f}" if "z" in row.index else "")
            + (f"<br>{row['event_name']}" if "event_name" in row.index else "")
            for _, row in dets_at_t.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=dets_at_t["x"].values,
            y=dets_at_t["y"].values,
            mode="markers",
            marker=dict(
                size=14,
                color="rgba(255, 0, 0, 0)",
                line=dict(color="red", width=2),
            ),
            text=hover_text,
            hoverinfo="text",
            name=f"Detections ({len(dets_at_t)})",
        ))

    fig.update_layout(
        title=dict(
            text=f"t={timepoint}, {len(dets_at_t)} detections",
            font=dict(size=12),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        dragmode="zoom",
        xaxis=dict(range=[0, w], showticklabels=False, scaleanchor="y"),
        yaxis=dict(range=[h, 0], showticklabels=False),
        showlegend=True,
        legend=dict(x=1, y=1, xanchor="right", bgcolor="rgba(255,255,255,0.7)"),
    )
    return fig


def _write_roi_config(job_uploads_dir):
    """Write centered ROI crop config JSON into the job uploads folder."""
    roi = {
        "x_size": st.session_state.get("roi_xw", 256),
        "y_size": st.session_state.get("roi_yw", 256),
        "z_size": st.session_state.get("roi_zw", 0),
    }
    roi_path = Path(job_uploads_dir) / "roi.json"
    with open(roi_path, "w") as f:
        json.dump(roi, f)
    log.info("ROI config: %s", roi)


def submit_to_jeanzay(job_id):
    """SSH to kapoorlabslogin node and sbatch the prediction job."""
    log.info("Submitting job %s via %s", job_id, SUBMIT_SCRIPT)
    result = subprocess.run(
        [str(SUBMIT_SCRIPT), job_id],
        capture_output=True,
        text=True,
        timeout=60,
    )
    log.info("SSH stdout: %s", result.stdout.strip())
    if result.stderr.strip():
        log.warning("SSH stderr: %s", result.stderr.strip())
    if result.returncode != 0:
        log.error("SSH/sbatch failed (rc=%d): %s", result.returncode, result.stderr.strip())
        raise RuntimeError(
            f"SSH/sbatch failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    output = result.stdout.strip()
    slurm_id = None
    for line in output.split("\n"):
        if "Submitted batch job" in line:
            slurm_id = line.strip().split()[-1]
    log.info("Job %s submitted, SLURM ID: %s", job_id, slurm_id)
    return slurm_id, output


def get_job_status(job_id):
    """Read status from results dir on the mount."""
    status_file = RESULTS_DIR / job_id / "status.txt"
    if not status_file.exists():
        return "queued"
    return status_file.read_text().strip()


def get_results(job_id):
    """Load results CSV from mount if available."""
    csv_path = RESULTS_DIR / job_id / "oneat_detections.csv"
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return pd.read_csv(csv_path), csv_path
    return None, None


def main():
    st.set_page_config(page_title="ONEAT Event Detection", layout="wide")

    st.title("ONEAT Event Detection")
    st.markdown(
        "Spatio-temporal event detection in 3D+T microscopy data — powered by KapoorLabsHPC"
    )

    # --- Sidebar: model selection ---
    st.sidebar.header("Model")
    available_models = discover_models()

    if available_models:
        model_name = st.sidebar.selectbox(
            "Select model",
            options=list(available_models.keys()),
            index=0,
        )
        model_info = available_models[model_name]
        st.sidebar.caption(f"Checkpoint: {Path(model_info['ckpt']).name}")
    else:
        st.sidebar.warning(
            f"No models found in {MODELS_DIR}. "
            "Place model directories (with .ckpt + .json) there."
        )

    # --- Sidebar: input files ---
    st.sidebar.header("Input Files")
    use_defaults = st.sidebar.checkbox("Use demo files", value=True)

    if use_defaults:
        raw_path = DEFAULT_RAW
        seg_path = DEFAULT_SEG
        has_defaults = raw_path.exists() and seg_path.exists()
        if has_defaults:
            st.sidebar.success(
                f"Raw: {raw_path.name}\nSeg: {seg_path.name}"
            )
        else:
            missing = []
            if not raw_path.exists():
                missing.append(str(raw_path))
            if not seg_path.exists():
                missing.append(str(seg_path))
            st.sidebar.error(
                "Default files not found:\n" + "\n".join(missing)
            )
        raw_file = None
        seg_file = None
    else:
        raw_file = st.sidebar.file_uploader(
            "Raw Timelapse (TIF)", type=["tif", "tiff"], key="raw"
        )
        seg_file = st.sidebar.file_uploader(
            "Segmentation Timelapse (TIF)", type=["tif", "tiff"], key="seg"
        )
        raw_path = None
        seg_path = None
        has_defaults = False

    # --- Sidebar: ROI crop (centered) ---
    st.sidebar.header("ROI Crop")
    st.sidebar.caption("Crop region centered on the image (smaller = faster demo)")
    roi_x_size = st.sidebar.number_input("X size", value=256, min_value=1, step=1, key="roi_xw")
    roi_y_size = st.sidebar.number_input("Y size", value=256, min_value=1, step=1, key="roi_yw")
    roi_z_size = st.sidebar.number_input("Z size (0 = full)", value=0, min_value=0, step=1, key="roi_zw")

    # --- Preview default images with T and Z sliders ---
    if use_defaults and has_defaults:
        st.subheader("Input Data Preview")

        # Show full image dimensions
        try:
            with TiffFile(str(raw_path)) as tif:
                full_shape = tif.series[0].shape
            if len(full_shape) >= 4:
                st.info(
                    f"Image size: T={full_shape[0]}, Z={full_shape[1]}, "
                    f"Y={full_shape[2]}, X={full_shape[3]}"
                )
            elif len(full_shape) == 3:
                st.info(f"Image size: Z={full_shape[0]}, Y={full_shape[1]}, X={full_shape[2]}")
        except Exception:
            pass

        try:
            with TiffFile(str(raw_path)) as tif:
                preview_shape = tif.series[0].shape
            if len(preview_shape) >= 4:
                preview_nt = preview_shape[0]
                preview_nz = preview_shape[1]
                preview_ny = preview_shape[2]
                preview_nx = preview_shape[3]
            elif len(preview_shape) == 3:
                preview_nt = 1
                preview_nz = preview_shape[0]
                preview_ny = preview_shape[1]
                preview_nx = preview_shape[2]
            else:
                preview_nt, preview_nz, preview_ny, preview_nx = 1, 1, 1, 1
        except Exception:
            preview_nt, preview_nz, preview_ny, preview_nx = 1, 1, 1, 1

        max_pt = max(preview_nt - 1, 0)
        max_pz = max(preview_nz - 1, 0)

        def _sync_preview(src, dst):
            st.session_state[dst] = st.session_state[src]

        if "prev_t_slider" not in st.session_state:
            st.session_state["prev_t_slider"] = 0
            st.session_state["prev_t_box"] = 0
        if "prev_z_slider" not in st.session_state:
            st.session_state["prev_z_slider"] = preview_nz // 2
            st.session_state["prev_z_box"] = preview_nz // 2

        col_t, col_z = st.columns(2)
        with col_t:
            st.slider(
                "Timepoint (preview)", 0, max_pt, step=1,
                key="prev_t_slider",
                on_change=_sync_preview, args=("prev_t_slider", "prev_t_box"),
            )
            st.number_input(
                "t", 0, max_pt,
                key="prev_t_box",
                on_change=_sync_preview, args=("prev_t_box", "prev_t_slider"),
            )
        with col_z:
            st.slider(
                "Z-slice (preview)", 0, max_pz, step=1,
                key="prev_z_slider",
                on_change=_sync_preview, args=("prev_z_slider", "prev_z_box"),
            )
            st.number_input(
                "z", 0, max_pz,
                key="prev_z_box",
                on_change=_sync_preview, args=("prev_z_box", "prev_z_slider"),
            )
        preview_t = st.session_state["prev_t_slider"]
        preview_z = st.session_state["prev_z_slider"]

        try:
            raw_slice, raw_shape = read_single_slice(raw_path, t=preview_t, z_mid=preview_z)
            fig = preview_slice_figure(
                raw_slice,
                f"Raw  ({raw_shape}, t={preview_t}, z={preview_z})",
            )
            h, w = raw_slice.shape[:2]
            cx, cy = w // 2, h // 2
            x0 = max(cx - roi_x_size // 2, 0)
            y0 = max(cy - roi_y_size // 2, 0)
            x1 = min(x0 + roi_x_size, w)
            y1 = min(y0 + roi_y_size, h)
            fig.add_shape(
                type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color="cyan", width=2, dash="dash"),
            )
            fig.add_annotation(
                x=x0, y=max(y0 - 5, 0),
                text=f"ROI {roi_x_size}x{roi_y_size}",
                showarrow=False, font=dict(color="cyan", size=11),
                xanchor="left",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Cannot preview raw: {e}")

        st.subheader("Download Demo Data")
        dl_raw, dl_seg = st.columns(2)
        with dl_raw:
            if raw_path.exists():
                with open(raw_path, "rb") as f:
                    st.download_button(
                        "Download Raw Timelapse",
                        data=f,
                        file_name=raw_path.name,
                        mime="image/tiff",
                    )
        with dl_seg:
            if seg_path.exists():
                with open(seg_path, "rb") as f:
                    st.download_button(
                        "Download Segmentation Timelapse",
                        data=f,
                        file_name=seg_path.name,
                        mime="image/tiff",
                    )

    # --- Submit button ---
    run_btn = st.button(
        "Run Prediction on KapoorLabs", type="primary", use_container_width=True
    )

    if run_btn:
        log.info("=== User clicked Run Prediction (use_defaults=%s) ===", use_defaults)
        if not SUBMIT_SCRIPT.exists():
            st.error(
                f"Submit script not found at {SUBMIT_SCRIPT}. "
                "Ensure the lustre mount is active."
            )
        elif use_defaults:
            if not has_defaults:
                st.error("Default demo files are missing from the mount.")
            else:
                job_id = uuid.uuid4().hex[:8]
                job_uploads = UPLOADS_DIR / job_id
                job_uploads.mkdir(parents=True, exist_ok=True)

                log.info("Job %s: linking demo files", job_id)
                with st.spinner("Linking demo files..."):
                    raw_dest = job_uploads / f"raw_{raw_path.name}"
                    seg_dest = job_uploads / f"seg_{seg_path.name}"
                    lustre_raw = LUSTRE_DEMO / "uploads" / raw_path.name
                    lustre_seg = LUSTRE_DEMO / "uploads" / seg_path.name
                    raw_dest.symlink_to(lustre_raw)
                    seg_dest.symlink_to(lustre_seg)
                    _write_roi_config(job_uploads)
                    st.session_state["raw_path_on_mount"] = str(raw_path)

                with st.spinner("SSH → kapoorlabs login node → sbatch..."):
                    try:
                        slurm_id, output = submit_to_jeanzay(job_id)
                        st.session_state["job_id"] = job_id
                        st.session_state["slurm_id"] = slurm_id
                        st.session_state.pop("results_df", None)
                        st.success(
                            f"Job submitted! ID: {job_id} (SLURM: {slurm_id})"
                        )
                    except Exception as e:
                        log.error("Job %s submission failed: %s", job_id, e)
                        st.error(f"Submission failed: {e}")
        else:
            if raw_file is None or seg_file is None:
                st.error("Upload both raw and segmentation timelapse files.")
            else:
                job_id = uuid.uuid4().hex[:8]
                job_uploads = UPLOADS_DIR / job_id
                job_uploads.mkdir(parents=True, exist_ok=True)

                log.info("Job %s: uploading user files (%s, %s)", job_id, raw_file.name, seg_file.name)
                with st.spinner("Writing files to KapoorLabs mount..."):
                    raw_dest = job_uploads / f"raw_{raw_file.name}"
                    seg_dest = job_uploads / f"seg_{seg_file.name}"
                    raw_dest.write_bytes(raw_file.read())
                    seg_dest.write_bytes(seg_file.read())
                    _write_roi_config(job_uploads)
                    st.session_state["raw_path_on_mount"] = str(raw_dest)

                with st.spinner("SSH → kapoorlabs login node → sbatch..."):
                    try:
                        slurm_id, output = submit_to_jeanzay(job_id)
                        st.session_state["job_id"] = job_id
                        st.session_state["slurm_id"] = slurm_id
                        st.session_state.pop("results_df", None)
                        st.success(
                            f"Job submitted! ID: {job_id} (SLURM: {slurm_id})"
                        )
                    except Exception as e:
                        log.error("Job %s submission failed: %s", job_id, e)
                        st.error(f"Submission failed: {e}")

    # --- Poll for status ---
    if "job_id" in st.session_state and "results_df" not in st.session_state:
        job_id = st.session_state["job_id"]
        slurm_id = st.session_state.get("slurm_id", "?")

        st.markdown("---")
        status = get_job_status(job_id)

        status_map = {
            "queued": "Waiting in SLURM queue...",
            "running": "Job starting...",
            "loading_model": "Loading model checkpoint...",
            "predicting": "Running inference on GPU...",
            "postprocessing": "Applying NMS...",
            "done": "Complete!",
        }

        if status.startswith("error"):
            log.error("Job %s failed: %s", job_id, status)
            st.error(f"Job failed: {status}")
        elif status == "done":
            log.info("Job %s complete!", job_id)
            st.success("Prediction complete!")
            df, csv_path = get_results(job_id)
            if df is not None and len(df) > 0:
                st.session_state["results_df"] = df
                st.session_state["csv_path"] = str(csv_path)
                st.rerun()
            else:
                st.warning("No events detected.")
        else:
            st.info(
                f"{status_map.get(status, status)}  (SLURM: {slurm_id})"
            )
            time.sleep(5)
            st.rerun()

    # --- Results display ---
    if "results_df" in st.session_state and len(st.session_state["results_df"]) > 0:
        df = st.session_state["results_df"]
        st.markdown("---")

        tab_table, tab_viewer = st.tabs(["Results Table", "Detection Viewer"])

        with tab_table:
            st.subheader(f"Detected Events ({len(df)})")

            if "event_name" in df.columns:
                event_counts = df["event_name"].value_counts()
                cols = st.columns(len(event_counts))
                for i, (event, count) in enumerate(event_counts.items()):
                    cols[i].metric(event, count)

            st.dataframe(df, use_container_width=True, hide_index=True)

            csv_path = st.session_state.get("csv_path")
            if csv_path and os.path.exists(csv_path):
                with open(csv_path, "rb") as f:
                    st.download_button(
                        "Download Predictions CSV",
                        data=f,
                        file_name="oneat_detections.csv",
                        mime="text/csv",
                    )

        with tab_viewer:
            st.subheader("Detection Overlay Viewer")
            raw_mount_path = st.session_state.get("raw_path_on_mount")

            if raw_mount_path and os.path.exists(raw_mount_path):
                with TiffFile(raw_mount_path) as tif:
                    shape = tif.series[0].shape
                    num_t = shape[0] if len(shape) >= 4 else 1
                    nz = shape[1] if len(shape) >= 4 else shape[0]
                    ny = shape[2] if len(shape) >= 4 else shape[1] if len(shape) >= 3 else 1
                    nx = shape[3] if len(shape) >= 4 else shape[2] if len(shape) >= 3 else 1

                viewer_crop = {"x_size": roi_x_size, "y_size": roi_y_size}
                cx_off = max(nx // 2 - roi_x_size // 2, 0)
                cy_off = max(ny // 2 - roi_y_size // 2, 0)
                st.caption(
                    f"Showing {roi_x_size}x{roi_y_size} crop centered on "
                    f"{nx}x{ny} image"
                )

                det_timepoints = sorted(df["t"].unique().tolist())

                if det_timepoints:
                    max_t = max(num_t - 1, 1)
                    max_z = max(nz - 1, 0)

                    def _sync(src, dst):
                        st.session_state[dst] = st.session_state[src]

                    if "det_t_slider" not in st.session_state:
                        st.session_state["det_t_slider"] = int(det_timepoints[0])
                        st.session_state["det_t_box"] = int(det_timepoints[0])
                    if "det_z_slider" not in st.session_state:
                        st.session_state["det_z_slider"] = nz // 2
                        st.session_state["det_z_box"] = nz // 2

                    col_dt, col_dz = st.columns(2)
                    with col_dt:
                        st.slider(
                            "Timepoint", 0, max_t, step=1,
                            key="det_t_slider",
                            on_change=_sync, args=("det_t_slider", "det_t_box"),
                        )
                        st.number_input(
                            "t", 0, max_t,
                            key="det_t_box",
                            on_change=_sync, args=("det_t_box", "det_t_slider"),
                        )
                    with col_dz:
                        st.slider(
                            "Z-slice", 0, max_z, step=1,
                            key="det_z_slider",
                            on_change=_sync, args=("det_z_slider", "det_z_box"),
                        )
                        st.number_input(
                            "z", 0, max_z,
                            key="det_z_box",
                            on_change=_sync, args=("det_z_box", "det_z_slider"),
                        )
                    selected_t = st.session_state["det_t_slider"]
                    selected_z = st.session_state["det_z_slider"]

                    n_at_t = len(df[df["t"] == selected_t])
                    st.caption(
                        f"t={selected_t}, z={selected_z} — {n_at_t} detections at this t  |  "
                        f"Timepoints with events: {det_timepoints}"
                    )

                    slice_2d, _ = read_single_slice(
                        raw_mount_path, t=selected_t, z_mid=selected_z,
                        crop=viewer_crop,
                    )
                    view_df = df.copy()
                    view_df["x"] = view_df["x"] - cx_off
                    view_df["y"] = view_df["y"] - cy_off
                    fig = create_detection_overlay(slice_2d, view_df, selected_t)
                    st.plotly_chart(fig, use_container_width=True)

                    dets_at_t = df[df["t"] == selected_t]
                    if len(dets_at_t) > 0:
                        st.dataframe(
                            dets_at_t,
                            use_container_width=True,
                            hide_index=True,
                        )
            else:
                st.warning("Raw image not available for overlay visualization.")


if __name__ == "__main__":
    main()

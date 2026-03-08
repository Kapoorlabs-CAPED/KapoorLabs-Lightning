"""
ONEAT Event Detection - Streamlit App

Upload raw + segmentation timelapse TIF files and run ONEAT prediction
on Jean Zay HPC via SSH + SLURM. Results appear via shared mount.

Usage:
    streamlit run app.py
"""

import os
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from tifffile import imread

# jean-zay lustre mounted locally via sshfs
JEANZAY_MOUNT = Path("/home/debian/jean-zay/demo")
UPLOADS_DIR = JEANZAY_MOUNT / "uploads"
RESULTS_DIR = JEANZAY_MOUNT / "results"

# SSH submit script — lives outside the repo at /home/debian/demo/
SUBMIT_SCRIPT = Path("/home/debian/demo/submit_job.sh")


def create_detection_overlay(raw_image, detections_df, timepoint):
    """Max-Z-projection with detection markers."""
    dets_at_t = detections_df[detections_df["t"] == timepoint]

    frame = raw_image[timepoint]
    max_proj = np.max(frame, axis=0)

    vmin, vmax = np.percentile(max_proj, (1, 99.8))
    if vmax > vmin:
        img = np.clip((max_proj - vmin) / (vmax - vmin), 0, 1)
    else:
        img = np.zeros_like(max_proj, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap="gray")

    if len(dets_at_t) > 0:
        ax.scatter(
            dets_at_t["x"].values,
            dets_at_t["y"].values,
            c="red",
            s=80,
            marker="o",
            facecolors="none",
            linewidths=2,
            label=f"Detections ({len(dets_at_t)})",
        )
        ax.legend(loc="upper right", fontsize=9)

    ax.set_title(f"Max-Z Projection   t={timepoint}   ({len(dets_at_t)} detections)")
    ax.axis("off")
    fig.tight_layout()
    return fig


def submit_to_jeanzay(job_id):
    """SSH to jean-zay login node and sbatch the prediction job."""
    result = subprocess.run(
        [str(SUBMIT_SCRIPT), job_id],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"SSH/sbatch failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    output = result.stdout.strip()
    slurm_id = None
    for line in output.split("\n"):
        if "Submitted batch job" in line:
            slurm_id = line.strip().split()[-1]
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
        "Spatio-temporal event detection in 3D+T microscopy data — powered by Jean Zay HPC"
    )

    # --- Sidebar: uploads ---
    st.sidebar.header("Input Files")
    raw_file = st.sidebar.file_uploader(
        "Raw Timelapse (TIF)", type=["tif", "tiff"], key="raw"
    )
    seg_file = st.sidebar.file_uploader(
        "Segmentation Timelapse (TIF)", type=["tif", "tiff"], key="seg"
    )
    st.sidebar.caption("Model architecture and NMS use training defaults.")

    # --- Submit button ---
    run_btn = st.button(
        "Run Prediction on Jean Zay", type="primary", use_container_width=True
    )

    if run_btn:
        if raw_file is None or seg_file is None:
            st.error("Upload both raw and segmentation timelapse files.")
        elif not SUBMIT_SCRIPT.exists():
            st.error(
                f"Submit script not found at {SUBMIT_SCRIPT}. "
                "Place submit_job.sh at /home/debian/demo/"
            )
        else:
            job_id = uuid.uuid4().hex[:8]
            job_uploads = UPLOADS_DIR / job_id
            job_uploads.mkdir(parents=True, exist_ok=True)

            with st.spinner("Writing files to Jean Zay mount..."):
                raw_dest = job_uploads / f"raw_{raw_file.name}"
                seg_dest = job_uploads / f"seg_{seg_file.name}"
                raw_dest.write_bytes(raw_file.read())
                seg_dest.write_bytes(seg_file.read())
                st.session_state["raw_path_on_mount"] = str(raw_dest)

            with st.spinner("SSH → jean-zay login node → sbatch..."):
                try:
                    slurm_id, output = submit_to_jeanzay(job_id)
                    st.session_state["job_id"] = job_id
                    st.session_state["slurm_id"] = slurm_id
                    # Clear old results
                    st.session_state.pop("results_df", None)
                    st.session_state.pop("raw_image", None)
                    st.success(f"Job submitted! ID: {job_id} (SLURM: {slurm_id})")
                except Exception as e:
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
            st.error(f"Job failed: {status}")
        elif status == "done":
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
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv_path = st.session_state.get("csv_path")
            if csv_path and os.path.exists(csv_path):
                with open(csv_path, "rb") as f:
                    st.download_button(
                        "Download CSV",
                        data=f,
                        file_name="oneat_detections.csv",
                        mime="text/csv",
                    )

        with tab_viewer:
            st.subheader("Detection Overlay Viewer")
            raw_mount_path = st.session_state.get("raw_path_on_mount")

            if raw_mount_path and os.path.exists(raw_mount_path):
                if "raw_image" not in st.session_state:
                    with st.spinner("Loading raw image for visualization..."):
                        st.session_state["raw_image"] = imread(raw_mount_path)

                raw_image = st.session_state["raw_image"]
                num_t = raw_image.shape[0]
                det_timepoints = sorted(df["t"].unique().tolist())

                if det_timepoints:
                    selected_t = st.slider(
                        "Timepoint",
                        min_value=0,
                        max_value=num_t - 1,
                        value=int(det_timepoints[0]),
                        step=1,
                    )

                    n_at_t = len(df[df["t"] == selected_t])
                    st.caption(
                        f"Timepoint {selected_t} — {n_at_t} detections  |  "
                        f"Timepoints with events: {det_timepoints}"
                    )

                    fig = create_detection_overlay(raw_image, df, selected_t)
                    st.pyplot(fig)
                    plt.close(fig)

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

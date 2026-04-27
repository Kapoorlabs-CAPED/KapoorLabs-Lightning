"""
ONEAT Event Detection - Streamlit App

Upload raw + segmentation timelapse TIF files and run ONEAT prediction
on KapoorLabs HPC via SSH + SLURM. Results appear via shared mount.

Usage:
    streamlit run remote_app.py
"""

import gc
import io
import json
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import urllib.error
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
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

# Per-demo metadata. Key = subdir name under UPLOADS_DIR. Anything not listed
# here falls back to a generic label so adding a new demo dir Just Works.
# `order` controls dropdown placement (lower = shown first).
DEMO_META = {
     "first_timepoint": {
        "label": "Early Timepoints (~1h runtime)",
        "blurb": (
            "Representative of the data we handle in production — full 3D+T "
            "Xenopus timelapse, multi-hour A100 inference."
        ),
        "order": 0,
    },
    "last_timepoint": {
        "label": "Later Timepoints (~3h runtime)",
        "blurb": (
            "Representative of the data we handle in production — full 3D+T "
            "Xenopus timelapse, multi-hour A100 inference."
        ),
        "order": 1,
    },
    
    "simple_data": {
        "label": "Toy demo (~15 min runtime)",
        "blurb": (
            "Small synthetic timelapse. Useful for a quick sanity check; not "
            "representative of real-world data sizes."
        ),
        "order": 2,
    },
}


def demo_meta(name):
    return DEMO_META.get(name, {"label": name, "blurb": None, "order": 99})


def discover_demos():
    """Scan UPLOADS_DIR for curated demo datasets.

    A demo is a subdirectory of UPLOADS_DIR containing exactly one
    `raw_*.tif/.tiff` and one `seg_*.tif/.tiff`. Returns a dict
    {display_name: {"raw": Path, "seg": Path}} sorted by name.
    """
    demos = {}
    if not UPLOADS_DIR.exists():
        return demos
    for entry in sorted(UPLOADS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        # Real files only — skip symlinks, which are how job dirs are wired.
        raws = sorted(
            p for p in entry.iterdir()
            if p.is_file() and not p.is_symlink()
            and p.name.lower().startswith("raw_")
            and p.suffix.lower() in (".tif", ".tiff")
        )
        segs = sorted(
            p for p in entry.iterdir()
            if p.is_file() and not p.is_symlink()
            and p.name.lower().startswith("seg_")
            and p.suffix.lower() in (".tif", ".tiff")
        )
        if raws and segs:
            demos[entry.name] = {"raw": raws[0], "seg": segs[0]}
    return demos

# Submit scripts on the lustre mount
SUBMIT_SCRIPT_FREE = JEANZAY_MOUNT / "submit_job.sh"
SUBMIT_SCRIPT_HEAVY = JEANZAY_MOUNT / "submit_heavy_job.sh"

# Accounting + quota
USAGE_LOG = JEANZAY_MOUNT / "usage_log.jsonl"

# ORCID-based gating
ORCID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")
HEAVY_QUOTA_PER_WEEK = 20
HEAVY_COOLDOWN_HOURS = 24


def heavy_quota_status(orcid):
    """Inspect usage log for `orcid`. Return (count_7d, last_heavy_ts_or_None)."""
    if not USAGE_LOG.exists() or not orcid:
        return 0, None
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    count = 0
    last_ts = None
    with USAGE_LOG.open() as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("orcid") != orcid or e.get("mode") != "heavy":
                continue
            if e.get("slurm_id") is None:
                continue  # only count successful submissions
            try:
                ts = datetime.fromisoformat(e["ts"].replace("Z", "+00:00"))
            except Exception:
                continue
            if ts < cutoff:
                continue
            count += 1
            if last_ts is None or ts > last_ts:
                last_ts = ts
    return count, last_ts


ORCID_AUTH_URL = "https://orcid.org/oauth/authorize"
ORCID_TOKEN_URL = "https://orcid.org/oauth/token"


def orcid_oauth_config():
    """Return (client_id, client_secret, redirect_uri) or (None, None, None) if unset."""
    try:
        cfg = st.secrets["orcid"]
        return cfg["client_id"], cfg["client_secret"], cfg["redirect_uri"]
    except Exception:
        return None, None, None


def build_orcid_login_url(client_id, redirect_uri):
    """Build the ORCID OAuth authorize URL with /authenticate scope."""
    params = {
        "client_id": client_id,
        "response_type": "code",
        "scope": "/authenticate",
        "redirect_uri": redirect_uri,
    }
    return f"{ORCID_AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_orcid_code(code, client_id, client_secret, redirect_uri):
    """Exchange auth code for an access token + verified ORCID iD.

    Returns the JSON dict (containing 'orcid', 'name', 'access_token', ...)
    or None on failure.
    """
    data = urllib.parse.urlencode({
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }).encode()
    req = urllib.request.Request(
        ORCID_TOKEN_URL,
        data=data,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        log.warning("ORCID token exchange failed: %s", e)
        return None


def heavy_embargo_until(orcid):
    """Return a datetime if the user is currently embargoed, else None."""
    count, last_ts = heavy_quota_status(orcid)
    if count >= HEAVY_QUOTA_PER_WEEK and last_ts is not None:
        embargo_end = last_ts + timedelta(hours=HEAVY_COOLDOWN_HOURS)
        if embargo_end > datetime.now(timezone.utc):
            return embargo_end
    return None


def get_client_ip():
    """Best-effort client IP from Streamlit request headers."""
    try:
        headers = st.context.headers
    except Exception:
        return "unknown"
    fwd = headers.get("X-Forwarded-For") or headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return headers.get("Host") or "unknown"


def log_usage(event):
    """Append a JSON line to the usage log. Never raises."""
    try:
        USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with USAGE_LOG.open("a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        log.warning("usage log write failed: %s", e)


def build_results_zip(paths):
    """Bundle existing files into an in-memory zip. Skips missing entries."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for arcname, src in paths.items():
            if src and Path(src).exists():
                zf.write(src, arcname=arcname)
    buf.seek(0)
    return buf.getvalue()


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


def read_single_slice(tif_path, t=0, z_mid=None):
    """Read a single Z-slice from a TZYX timelapse via page-level access."""
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
        return np.asarray(tif.pages[page_idx].asarray()), shape


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
        # Keep this constant across reruns so Plotly preserves the user's
        # zoom/pan when t/z changes or results arrive.
        uirevision="oneat-viewer",
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

def preview_seg_slice_figure(img_2d, title=""):
    """Interactive plotly figure with zoom/pan from a 2D array."""
    img8 = img_2d.astype(np.uint16)
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


def render_image_viewer(raw_image_path, results_df=None):
    """One viewer to rule them both — pre-submit preview and post-submit overlay.

    If `results_df` is provided and non-empty, detection markers and
    prev/next-event buttons are layered onto the slice. Otherwise the
    same widgets render a plain preview.

    Reads slices via tifffile page-level access (one page at a time, no
    full-volume load), so this is cheap to call on every Streamlit rerun.
    """
    if not raw_image_path or not os.path.exists(raw_image_path):
        return

    has_results = results_df is not None and len(results_df) > 0

    try:
        with TiffFile(str(raw_image_path)) as tif:
            shape = tif.series[0].shape
        if len(shape) >= 4:
            num_t, num_z = shape[0], shape[1]
        elif len(shape) == 3:
            num_t, num_z = 1, shape[0]
        else:
            num_t, num_z = 1, 1
    except Exception as e:
        st.warning(f"Cannot read raw image: {e}")
        return

    max_t = max(num_t - 1, 0)
    max_z = max(num_z - 1, 0)

    det_timepoints = sorted(results_df["t"].unique().tolist()) if has_results else []
    initial_t = int(det_timepoints[0]) if det_timepoints else 0

    def _sync(src, dst):
        st.session_state[dst] = st.session_state[src]

    # Apply any pending snap target stashed by the polling loop on the
    # previous run (must happen BEFORE widget instantiation).
    pending_t = st.session_state.pop("_pending_snap_t", None)
    if pending_t is not None:
        st.session_state["view_t_slider"] = int(pending_t)
        st.session_state["view_t_box"] = int(pending_t)

    if "view_t_slider" not in st.session_state:
        st.session_state["view_t_slider"] = initial_t
        st.session_state["view_t_box"] = initial_t
    if "view_z_slider" not in st.session_state:
        st.session_state["view_z_slider"] = num_z // 2
        st.session_state["view_z_box"] = num_z // 2

    def _jump_event(direction):
        if not det_timepoints:
            return
        cur = st.session_state.get("view_t_slider", det_timepoints[0])
        if direction == "next":
            cands = [t for t in det_timepoints if t > cur]
            new_t = cands[0] if cands else det_timepoints[0]
        else:
            cands = [t for t in det_timepoints if t < cur]
            new_t = cands[-1] if cands else det_timepoints[-1]
        st.session_state["view_t_slider"] = int(new_t)
        st.session_state["view_t_box"] = int(new_t)

    col_t, col_z = st.columns(2)
    with col_t:
        if has_results:
            b_prev, b_next = st.columns(2)
            b_prev.button(
                "< Prev event", on_click=_jump_event, args=("prev",),
                use_container_width=True, key="view_btn_prev",
                help="Jump to the previous timepoint with detections (wraps).",
            )
            b_next.button(
                "Next event >", on_click=_jump_event, args=("next",),
                use_container_width=True, key="view_btn_next",
                help="Jump to the next timepoint with detections (wraps).",
            )
        st.slider(
            "Timepoint", 0, max_t, step=1,
            key="view_t_slider",
            on_change=_sync, args=("view_t_slider", "view_t_box"),
        )
        st.number_input(
            "t", 0, max_t,
            key="view_t_box",
            on_change=_sync, args=("view_t_box", "view_t_slider"),
        )
    with col_z:
        st.slider(
            "Z-slice", 0, max_z, step=1,
            key="view_z_slider",
            on_change=_sync, args=("view_z_slider", "view_z_box"),
        )
        st.number_input(
            "z", 0, max_z,
            key="view_z_box",
            on_change=_sync, args=("view_z_box", "view_z_slider"),
        )
    selected_t = st.session_state["view_t_slider"]
    selected_z = st.session_state["view_z_slider"]

    if has_results:
        n_at_t = len(results_df[results_df["t"] == selected_t])
        st.caption(
            f"t={selected_t}, z={selected_z} — {n_at_t} detections at this t  |  "
            f"Timepoints with events: {det_timepoints}"
        )
    else:
        st.caption(f"t={selected_t}, z={selected_z}")

    try:
        slice_2d, raw_shape = read_single_slice(
            raw_image_path, t=selected_t, z_mid=selected_z
        )
    except Exception as e:
        st.warning(f"Cannot read slice: {e}")
        return

    if has_results:
        fig = create_detection_overlay(slice_2d, results_df, selected_t)
    else:
        fig = preview_slice_figure(
            slice_2d, f"Raw  ({raw_shape}, t={selected_t}, z={selected_z})"
        )

    # Click-to-zoom: if a previous click stashed a centre, override the
    # axis ranges to a small box around it. Bump uirevision to a unique
    # value so Plotly applies the new range instead of preserving the
    # previous zoom (uirevision is what makes pan/scroll-zoom sticky).
    h, w = slice_2d.shape
    zoom_half = max(64, min(h, w) // 16)  # ~1/8 of the smaller image dim
    zoom_center = st.session_state.get("zoom_center")
    if zoom_center is not None:
        cx, cy = zoom_center
        cx = max(zoom_half, min(w - zoom_half, cx))
        cy = max(zoom_half, min(h - zoom_half, cy))
        fig.update_layout(
            xaxis=dict(range=[cx - zoom_half, cx + zoom_half],
                       uirevision=f"zoom-{cx}-{cy}"),
            yaxis=dict(range=[cy + zoom_half, cy - zoom_half],
                       uirevision=f"zoom-{cx}-{cy}"),
            uirevision=f"zoom-{cx}-{cy}",
        )

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        key="oneat_image_viewer",
        on_select="rerun",
        selection_mode=("points",),
    )

    # Click captured via on_select returns selected points; first one wins.
    points = []
    if isinstance(event, dict):
        points = (event.get("selection") or {}).get("points") or []
    if points:
        p = points[0]
        new_cx = p.get("x")
        new_cy = p.get("y")
        if new_cx is not None and new_cy is not None:
            new_pair = (float(new_cx), float(new_cy))
            if st.session_state.get("zoom_center") != new_pair:
                st.session_state["zoom_center"] = new_pair
                st.rerun()

    if st.session_state.get("zoom_center") is not None:
        if st.button("Reset zoom", key="reset_zoom_btn"):
            st.session_state.pop("zoom_center", None)
            st.rerun()

    if has_results:
        dets_at_t = results_df[results_df["t"] == selected_t]
        if len(dets_at_t) > 0:
            st.dataframe(dets_at_t, use_container_width=True, hide_index=True)


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
        xaxis=dict(
            range=[0, w], showticklabels=False, scaleanchor="y",
            uirevision="oneat-viewer",
        ),
        yaxis=dict(
            range=[h, 0], showticklabels=False,
            uirevision="oneat-viewer",
        ),
        showlegend=True,
        legend=dict(x=1, y=1, xanchor="right", bgcolor="rgba(255,255,255,0.7)"),
        # Stable across reruns → Plotly preserves zoom/pan/legend toggles.
        uirevision="oneat-viewer",
    )
    return fig


def mount_to_lustre(mount_path):
    """Translate a path on the local sshfs mount to its lustre equivalent."""
    rel = Path(mount_path).resolve().relative_to(JEANZAY_MOUNT.resolve())
    return str(LUSTRE_DEMO / rel)


def submit_to_jeanzay(submit_script, job_id, ckpt_lustre, config_lustre=None):
    """SSH to kapoorlabslogin node and sbatch the prediction job."""
    log.info(
        "Submitting job %s via %s ckpt=%s config=%s",
        job_id, submit_script.name, ckpt_lustre, config_lustre,
    )
    cmd = [str(submit_script), job_id, ckpt_lustre]
    if config_lustre:
        cmd.append(config_lustre)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired as e:
        log.error(
            "SSH submit timed out after %ss. partial stdout=%r stderr=%r",
            e.timeout,
            (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or ""),
            (e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, (bytes, bytearray)) else (e.stderr or ""),
        )
        raise
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
        "Spatio-temporal event detection in 3D+T microscopy data — powered by KapoorLabs HPC"
    )

    # --- Sidebar: model selection ---
    st.sidebar.header("Model")
    available_models = discover_models()

    model_info = None
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

    # --- Sidebar: identity (ORCID OAuth-gated A100 access + usage accounting) ---
    st.sidebar.header("User (ORCID)")
    client_id, client_secret, redirect_uri = orcid_oauth_config()

    # Handle the OAuth callback: ORCID redirects back with ?code=...
    qp = st.query_params
    code = qp.get("code")
    if code and not st.session_state.get("verified_orcid") and client_id:
        with st.spinner("Verifying ORCID sign-in..."):
            tok = exchange_orcid_code(code, client_id, client_secret, redirect_uri)
        if tok and tok.get("orcid"):
            st.session_state["verified_orcid"] = tok["orcid"]
            st.session_state["verified_name"] = tok.get("name")
            log.info("ORCID OAuth success for %s", tok["orcid"])
        else:
            st.sidebar.error("ORCID sign-in failed. Please try again.")
        # Clear ?code from URL so a refresh doesn't re-exchange it
        st.query_params.clear()

    verified_orcid = st.session_state.get("verified_orcid")
    verified_name = st.session_state.get("verified_name")

    if not client_id:
        st.sidebar.error(
            "ORCID OAuth not configured. Add `[orcid]` block to "
            ".streamlit/secrets.toml."
        )
    elif verified_orcid:
        st.sidebar.success(
            f"Signed in as {verified_name or verified_orcid}\n\n`{verified_orcid}`"
        )
        if st.sidebar.button("Sign out", use_container_width=True):
            st.session_state.pop("verified_orcid", None)
            st.session_state.pop("verified_name", None)
            st.rerun()
    else:
        login_url = build_orcid_login_url(client_id, redirect_uri)
        st.sidebar.markdown(
            f"<a href='{login_url}' target='_self'>"
            f"<button style='width:100%;padding:0.5em;border-radius:0.5em;"
            f"border:1px solid #A6CE39;background:#A6CE39;color:white;"
            f"cursor:pointer;font-weight:600;'>Sign in with ORCID</button></a>",
            unsafe_allow_html=True,
        )
        st.sidebar.caption("Required for using our GPUs.")

    embargo_end = heavy_embargo_until(verified_orcid) if verified_orcid else None
    runs_7d = heavy_quota_status(verified_orcid)[0] if verified_orcid else 0
    is_heavy_allowed = bool(verified_orcid) and embargo_end is None

    if verified_orcid:
        st.sidebar.caption(
            f"Heavy runs in last 7 days: {runs_7d} / {HEAVY_QUOTA_PER_WEEK}"
        )
        if embargo_end is not None:
            st.sidebar.warning(
                f"Quota exceeded. Embargo until "
                f"{embargo_end.strftime('%Y-%m-%d %H:%M UTC')}."
            )
        else:
            st.sidebar.success("A100 access enabled.")

    # --- Sidebar: input files ---
    st.sidebar.header("Input Files")
    available_demos = discover_demos()
    # Sort by configured order, unknown demos fall to the bottom.
    demo_names = sorted(
        available_demos.keys(),
        key=lambda n: (demo_meta(n)["order"], n),
    )

    # Display label includes the runtime hint from DEMO_META.
    label_to_name = {f"Demo — {demo_meta(n)['label']}": n for n in demo_names}
    options = list(label_to_name.keys()) + ["Upload my own"]
    if not demo_names:
        options = ["Upload my own"]
    choice = st.sidebar.selectbox("Source", options, index=0)
    use_defaults = choice in label_to_name

    raw_file = None
    seg_file = None
    raw_path = None
    seg_path = None
    has_defaults = False
    demo_name = None

    if use_defaults:
        demo_name = label_to_name[choice]
        demo = available_demos[demo_name]
        raw_path = demo["raw"]
        seg_path = demo["seg"]
        has_defaults = raw_path.exists() and seg_path.exists()
        blurb = demo_meta(demo_name)["blurb"]
        if blurb:
            st.sidebar.caption(blurb)
        if has_defaults:
            st.sidebar.success(
                f"Raw: {raw_path.name}\nSeg: {seg_path.name}"
            )
        else:
            st.sidebar.error(
                f"Demo '{demo_name}' files missing on the mount."
            )
    else:
        if not verified_orcid:
            st.sidebar.warning(
                "Sign in with ORCID above to enable file uploads."
            )
        raw_file = st.sidebar.file_uploader(
            "Raw Timelapse (TIF)",
            type=["tif", "tiff"],
            key="raw",
            disabled=not verified_orcid,
        )
        seg_file = st.sidebar.file_uploader(
            "Segmentation Timelapse (TIF)",
            type=["tif", "tiff"],
            key="seg",
            disabled=not verified_orcid,
        )

    # --- Sidebar footer: project links ---
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**KapoorLabs / ONEAT**  \n"
        "[GitHub repo](https://github.com/Kapoorlabs-CAPED/KapoorLabs-Lightning) · "
        "[Napari plugin]"
        "(https://github.com/Kapoorlabs-CAPED/KapoorLabs-Lightning"
        "/blob/main/plugins/oneat_event_visualizer.py)  \n"
        "`pip install KapoorLabs-Lightning`"
    )

    # --- Unified image viewer ---
    # Same viewer is used pre-submit (preview) and post-results (overlay).
    # We pick the most up-to-date raw path: post-submit session state if
    # available, otherwise the demo raw_path for pre-submit preview.
    viewer_raw = (
        st.session_state.get("raw_path_on_mount")
        or (str(raw_path) if (use_defaults and has_defaults) else None)
    )
    viewer_df = st.session_state.get("results_df")
    if viewer_raw:
        st.subheader(
            "Detection Overlay" if (viewer_df is not None and len(viewer_df) > 0)
            else "Input Data Preview"
        )
        render_image_viewer(viewer_raw, results_df=viewer_df)


    # --- Submit: single ORCID-gated A100 button ---
    # Paint it green when unlocked so the run-state is obvious.
    if is_heavy_allowed:
        st.markdown(
            "<style>"
            "div.st-key-run_btn button[kind='primary'] {"
            "  background-color: #22c55e !important;"
            "  border-color: #16a34a !important;"
            "  color: white !important;"
            "}"
            "div.st-key-run_btn button[kind='primary']:hover {"
            "  background-color: #16a34a !important;"
            "  border-color: #15803d !important;"
            "}"
            "</style>",
            unsafe_allow_html=True,
        )

    if not verified_orcid:
        st.info(
            "Sign in with ORCID in the sidebar to submit a job. "
            "Every submission runs on KapoorLabs A100 GPUs and is logged "
            "against your ORCID for fair-use accounting."
        )

    run_btn = st.button(
        "Run on KapoorLabs A100",
        type="primary" if is_heavy_allowed else "secondary",
        use_container_width=True,
        disabled=not is_heavy_allowed,
        key="run_btn",
        help=None if is_heavy_allowed
        else "Sign in with ORCID and stay within quota.",
    )

    if run_btn:
        mode = "heavy"
        submit_script = SUBMIT_SCRIPT_HEAVY
        client_ip = get_client_ip()
        log.info(
            "=== Run clicked use_defaults=%s orcid=%s ip=%s ===",
            use_defaults, verified_orcid or "<none>", client_ip,
        )

        # Re-check embargo at click time — log on disk is the source of truth
        recheck = heavy_embargo_until(verified_orcid)
        if recheck is not None:
            st.error(
                f"Quota exceeded. Embargo until "
                f"{recheck.strftime('%Y-%m-%d %H:%M UTC')}."
            )
            st.stop()

        if not submit_script.exists():
            st.error(
                f"Submit script not found at {submit_script}. "
                "Ensure the lustre mount is active."
            )
        elif use_defaults and not has_defaults:
            st.error("Default demo files are missing from the mount.")
        elif not use_defaults and (raw_file is None or seg_file is None):
            st.error("Upload both raw and segmentation timelapse files.")
        elif model_info is None:
            st.error("No model selected.")
        else:
            job_id = uuid.uuid4().hex[:8]
            job_uploads = UPLOADS_DIR / job_id
            job_uploads.mkdir(parents=True, exist_ok=True)

            if use_defaults:
                with st.spinner("Linking demo files..."):
                    # raw_path lives at UPLOADS_DIR/<demo_name>/<filename>;
                    # the lustre equivalent is LUSTRE_DEMO/uploads/<demo_name>/<filename>.
                    lustre_demo_dir = LUSTRE_DEMO / "uploads" / demo_name
                    raw_dest = job_uploads / f"raw_{raw_path.name}"
                    seg_dest = job_uploads / f"seg_{seg_path.name}"
                    raw_dest.symlink_to(lustre_demo_dir / raw_path.name)
                    seg_dest.symlink_to(lustre_demo_dir / seg_path.name)
                    st.session_state["raw_path_on_mount"] = str(raw_path)
                    st.session_state["seg_path_on_mount"] = str(seg_path)
            else:
                # Browser uploads: stream each UploadedFile to the local mount
                # in 8 MiB chunks (low write-side RAM). Per-user, per-job
                # subdir keyed by ORCID + job_id keeps concurrent users isolated.
                user_dir = UPLOADS_DIR / verified_orcid / job_id
                user_dir.mkdir(parents=True, exist_ok=True)
                raw_dest = user_dir / f"raw_{raw_file.name}"
                seg_dest = user_dir / f"seg_{seg_file.name}"

                with st.spinner("Writing raw to mount..."):
                    raw_file.seek(0)
                    with open(raw_dest, "wb") as fh:
                        shutil.copyfileobj(raw_file, fh, length=8 * 1024 * 1024)
                    raw_file.close()
                    raw_file = None
                    gc.collect()

                with st.spinner("Writing seg to mount..."):
                    seg_file.seek(0)
                    with open(seg_dest, "wb") as fh:
                        shutil.copyfileobj(seg_file, fh, length=8 * 1024 * 1024)
                    seg_file.close()
                    seg_file = None
                    gc.collect()

                # Compute-node lustre symlinks so sbatch resolves the paths.
                lustre_user_dir = (
                    LUSTRE_DEMO / "uploads" / verified_orcid / job_id
                )
                # job_uploads dir already created above; symlinks live there.
                (job_uploads / raw_dest.name).symlink_to(
                    lustre_user_dir / raw_dest.name
                )
                (job_uploads / seg_dest.name).symlink_to(
                    lustre_user_dir / seg_dest.name
                )

                st.session_state["raw_path_on_mount"] = str(raw_dest)
                st.session_state["seg_path_on_mount"] = str(seg_dest)

            ckpt_lustre = mount_to_lustre(model_info["ckpt"])
            config_lustre = (
                mount_to_lustre(model_info["training_json"])
                if model_info.get("training_json") else None
            )

            slurm_id = None
            err = None
            with st.spinner("SSH → kapoorlabs login node → sbatch..."):
                try:
                    slurm_id, _ = submit_to_jeanzay(
                        submit_script, job_id, ckpt_lustre, config_lustre
                    )
                    st.session_state["job_id"] = job_id
                    st.session_state["slurm_id"] = slurm_id
                    st.session_state["used_defaults"] = use_defaults
                    st.session_state.pop("results_df", None)
                    st.success(f"Job submitted! ID: {job_id} (SLURM: {slurm_id})")
                except Exception as e:
                    err = str(e)
                    log.error("Job %s submission failed: %s", job_id, e)
                    st.error(f"Submission failed: {e}")

            log_usage({
                "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "ip": client_ip,
                "orcid": verified_orcid,
                "job_id": job_id,
                "slurm_id": slurm_id,
                "mode": mode,
                "model": model_info["dir"],
                "use_defaults": use_defaults,
                "error": err,
            })

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
                # Stash a snap-target; the viewer applies it BEFORE
                # instantiating widgets on the next rerun. Setting
                # view_t_slider directly here would crash because the
                # widget was already created earlier in this run.
                first_t = int(sorted(df["t"].unique().tolist())[0])
                st.session_state["_pending_snap_t"] = first_t
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

        # Image + overlay are rendered by the unified viewer up the page;
        # this block now only handles the table + downloads + plugin pointer.
        st.subheader(f"Detected Events ({len(df)})")
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_path = st.session_state.get("csv_path")
        raw_mount_path = st.session_state.get("raw_path_on_mount")
        seg_mount_path = st.session_state.get("seg_path_on_mount")
        job_id = st.session_state.get("job_id", "results")

        col_csv, col_zip = st.columns(2)

        if csv_path and os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                col_csv.download_button(
                    "Download CSV",
                    data=f,
                    file_name="oneat_detections.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        # Always offer the bundled zip — users (whether they used a demo or
        # uploaded) may be viewing results from a different machine than the
        # one they uploaded from, so they need raw + seg + CSV in one place.
        zip_cache_key = f"results_zip::{job_id}"
        zip_bytes = st.session_state.get(zip_cache_key)
        if zip_bytes is None and (csv_path or raw_mount_path or seg_mount_path):
            zip_bytes = build_results_zip({
                "oneat_detections.csv": csv_path,
                (f"raw_{Path(raw_mount_path).name}" if raw_mount_path else "raw"):
                    raw_mount_path,
                (f"seg_{Path(seg_mount_path).name}" if seg_mount_path else "seg"):
                    seg_mount_path,
            })
            st.session_state[zip_cache_key] = zip_bytes

        if zip_bytes:
            col_zip.download_button(
                "Download all (raw + seg + CSV)",
                data=zip_bytes,
                file_name=f"oneat_results_{job_id}.zip",
                mime="application/zip",
                use_container_width=True,
                help="Open the raw + seg TIFs in napari or Fiji for 3D inspection.",
            )

        with st.expander("Want full 3D inspection? Use our napari plugin"):
            st.markdown(
                "We ship a napari plugin that loads the raw + seg + "
                "detections CSV and lets you scrub through events in true 3D, "
                "edit annotations, and re-export.\n\n"
                "**Install:**\n"
                "```bash\n"
                "pip install KapoorLabs-Lightning napari\n"
                "```\n\n"
                "**Run** the plugin script directly from the repo:\n"
                "[oneat_event_visualizer.py on GitHub]"
                "(https://github.com/Kapoorlabs-CAPED/KapoorLabs-Lightning"
                "/blob/main/plugins/oneat_event_visualizer.py)\n\n"
                "Point it at the unzipped folder you just downloaded."
            )


if __name__ == "__main__":
    main()
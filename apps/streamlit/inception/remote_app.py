"""Inception Cell-Fate Prediction — Streamlit App

User picks a curated demo dataset + an Inception checkpoint + tracklet
length, and the app submits a SLURM job on Jean Zay to run
:mod:`predict-cellfate` against the demo's TrackMate XML. No user
uploads — every TIFF + XML + GT CSV lives on the Lustre share under
``demo_inception/uploads/<demo_name>/``.

When the job finishes:

* Per-class predicted track-ID CSVs are pulled from the shared mount.
* Ground-truth CSVs under ``<demo>/ground_truth/<class>.csv`` are
  matched against the predictions by ``Track_ID`` to build a confusion
  matrix (sklearn-shape, displayed as a plotly heatmap).
* The companion timelapse TIFF is rendered with a lazy frame reader —
  only the current timepoint is in memory at any moment — and per-cell
  GT vs predicted fate overlays are drawn on the max-Z projection at
  each track's centroid.

Usage:
    streamlit run remote_app.py
"""

# ---------------------------------------------------------------- imports
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xml.etree.ElementTree as ET
from scipy.spatial import cKDTree
from sklearn.metrics import confusion_matrix
from tifffile import TiffFile

# ---------------------------------------------------------------- feature plots
# Columns in the per-frame features csv (written by TrackVectors during
# predict-cellfate's feature-extraction step) that we expose in the
# GT-vs-prediction comparison panel. Each is aggregated per track
# (mean across all frames the track exists in) before plotting.
FEATURE_COLUMNS = [
    "Radius",
    "Eccentricity_Comp_First",
    "Eccentricity_Comp_Second",
    "Eccentricity_Comp_Third",
    "Surface_Area",
    "Local_Cell_Density",
    "Speed",
    "Acceleration",
    "MSD",
    "Track_Displacement",
    "Total_Track_Distance",
    "Max_Track_Distance",
    "Track_Duration",
    "Total_Intensity",
    "Mean_Intensity",
]
DEFAULT_FEATURE_PICKS = [
    "Radius",
    "Eccentricity_Comp_First",
    "Eccentricity_Comp_Second",
    "Surface_Area",
    "Speed",
    "MSD",
]

# ---------------------------------------------------------------- viewer
# Class → colour for GT / prediction overlays. Same palette is used
# across the streamlit + matplotlib code so labels in the legend match
# the dots on the image.
CLASS_COLORS = {
    "basal": "#33ff66",
    "goblet": "#33ccff",
    "radial": "#ff9933",
    "radially_intercalating": "#ff9933",
}


def _class_color(cls: str) -> str:
    """Lookup a colour for ``cls`` (case-insensitive). Falls back to
    matplotlib's tab10 first entry for unknown classes so nothing
    silently disappears."""
    key = (cls or "").lower()
    for k, v in CLASS_COLORS.items():
        if k in key:
            return v
    return "#cccccc"


def _read_calibration(xml_path: Path) -> tuple[float, float, float]:
    """Parse ``(dz, dy, dx)`` (in µm/voxel) from a TrackMate-style XML.
    Looks for ``voxeldepth`` / ``pixelheight`` / ``pixelwidth`` on the
    ``ImageData`` element (case-insensitive attribute matching since
    TrackMate writes them lowercase). Returns ``(1.0, 1.0, 1.0)`` if
    the XML isn't readable — overlays then render in CSV-native units."""
    try:
        root = ET.parse(xml_path).getroot()
    except Exception as e:
        log.warning("cannot parse calibration from %s: %s", xml_path, e)
        return (1.0, 1.0, 1.0)
    dz = dy = dx = 1.0
    for elem in root.iter():
        attrs = {k.lower(): v for k, v in elem.attrib.items()}
        if "voxeldepth" in attrs:
            try:
                dz = float(attrs["voxeldepth"])
            except ValueError:
                pass
        if "pixelheight" in attrs:
            try:
                dy = float(attrs["pixelheight"])
            except ValueError:
                pass
        if "pixelwidth" in attrs:
            try:
                dx = float(attrs["pixelwidth"])
            except ValueError:
                pass
    return (dz or 1.0, dy or 1.0, dx or 1.0)


# Per-class GT csv filename → class label. The GT files predict-cellfate
# expects are fixed names; map them so the viewer legend matches the
# confusion-matrix rows.
_GT_FILENAME_TO_CLASS = {
    "basal_cells_nuclei_annotations.csv": "basal",
    "goblet_cells_nuclei_annotations.csv": "goblet",
    "radially_intercalating_cells_nuclei_annotations.csv": "radial",
}


def _load_positions(csv_path: Path, cls: str) -> pd.DataFrame | None:
    """Read a position CSV (GT uses ``T,Z,Y,X``; per-class predictions
    use ``t,z,y,x``). Coords are already in pixel units (verified
    against the demo's 1560×1560 TIFF: GT y ∈ [125, 1418] fits the
    image extent — no calibration division is needed). Returns a
    DataFrame with columns ``t, z, y, x, class`` or ``None`` if the
    file can't be parsed."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        log.warning("cannot read %s: %s", csv_path, e)
        return None
    cols_lower = {c.lower(): c for c in df.columns}
    needed = ("t", "z", "y", "x")
    if not all(n in cols_lower for n in needed):
        return None
    df = df.rename(columns={cols_lower[n]: n for n in needed})
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(needed))
    df["class"] = cls
    return df[["t", "z", "y", "x", "class"]]


@st.cache_data(show_spinner=False)
def _load_features_aggregated(features_csv: str) -> pd.DataFrame:
    """Load the per-track-per-frame features csv and aggregate to ONE
    row per track (mean of every numeric column). Cached per file path
    so the 100 MB features csv is parsed once per session. Each row's
    ``TrackMate_Track_ID`` is the parent track id used as the join key
    against GT assignments and prediction labels."""
    feat = pd.read_csv(features_csv)
    if "TrackMate_Track_ID" not in feat.columns:
        log.warning(
            "features csv missing TrackMate_Track_ID (got %s)", list(feat.columns)
        )
        return pd.DataFrame(columns=["TrackMate_Track_ID"])
    num_cols = [
        c for c in feat.select_dtypes(include="number").columns
        if c != "TrackMate_Track_ID"
    ]
    agg = (
        feat.groupby("TrackMate_Track_ID", as_index=False)[num_cols]
        .mean()
    )
    return agg


@st.cache_data(show_spinner=False)
def _match_gt_to_tracks(
    features_csv: str, gt_csv_paths: tuple[str, ...]
) -> pd.DataFrame:
    """Local fallback for ``<prefix>gt_track_assignments.csv``: redo the
    nearest-neighbour spatial match between each GT (T,Z,Y,X) and the
    features csv's track rows. Returns a DataFrame with columns
    ``TrackMate_Track_ID, GT_Class``; conflicting tracks (matched to
    more than one GT class) are dropped to mirror what predict-cellfate
    does server-side."""
    feat = pd.read_csv(
        features_csv,
        usecols=["TrackMate_Track_ID", "t", "z", "y", "x"],
    )
    rows = []
    for gt_str in gt_csv_paths:
        gt_path = Path(gt_str)
        cls_key = _GT_FILENAME_TO_CLASS.get(
            gt_path.name, gt_path.stem.split("_")[0]
        )
        cls = cls_key.capitalize()
        try:
            gt = pd.read_csv(gt_path)
        except Exception as e:
            log.warning("cannot read GT csv %s: %s", gt_path, e)
            continue
        cols = {c.lower(): c for c in gt.columns}
        if not all(k in cols for k in ("t", "z", "y", "x")):
            continue
        T = gt[cols["t"]].astype(int).values
        Z = gt[cols["z"]].astype(float).values
        Y = gt[cols["y"]].astype(float).values
        X = gt[cols["x"]].astype(float).values
        for t_val in np.unique(T):
            frame_df = feat[feat["t"].round().astype(int) == int(t_val)]
            if frame_df.empty:
                continue
            tree = cKDTree(frame_df[["z", "y", "x"]].values)
            mask = T == t_val
            q = np.column_stack([Z[mask], Y[mask], X[mask]])
            _, idx = tree.query(q)
            for tid in frame_df.iloc[idx]["TrackMate_Track_ID"].tolist():
                rows.append({"TrackMate_Track_ID": tid, "GT_Class": cls})
    if not rows:
        return pd.DataFrame(columns=["TrackMate_Track_ID", "GT_Class"])
    df = pd.DataFrame(rows).drop_duplicates()
    # Drop tracks that picked up more than one GT class (genuine
    # spatial conflict — same as evaluate_against_gt does).
    n_per_tid = df.groupby("TrackMate_Track_ID")["GT_Class"].nunique()
    safe = n_per_tid[n_per_tid == 1].index
    return df[df["TrackMate_Track_ID"].isin(safe)].reset_index(drop=True)


def _violin_by_class(
    df: pd.DataFrame, feature: str, class_col: str, title: str
) -> plt.Figure:
    """Violin plot of one feature grouped by class. Returns the
    matplotlib Figure so the caller can pass it to ``st.pyplot`` then
    close it. Uses the per-class colour palette so GT and prediction
    panels are visually consistent."""
    classes = sorted(df[class_col].dropna().astype(str).unique())
    data = [df.loc[df[class_col] == c, feature].dropna().values for c in classes]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    nonempty = [d for d in data if len(d) > 0]
    if nonempty:
        parts = ax.violinplot(
            data,
            positions=range(len(classes)),
            showmeans=True,
            showmedians=False,
            widths=0.7,
        )
        for body, cls_label in zip(parts["bodies"], classes):
            body.set_facecolor(_class_color(cls_label.lower()))
            body.set_edgecolor("black")
            body.set_alpha(0.55)
        for key in ("cmeans", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color("black")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(
            [f"{c}\n(n={len(d)})" for c, d in zip(classes, data)],
            fontsize=8,
        )
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_xticks([])
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(feature, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def _load_per_frame_predictions(
    features_csv: str,
    all_predictions_csv: str,
) -> pd.DataFrame:
    """Per-track features csv (one row per (track, frame) — written by
    TrackVectors during predict-cellfate's feature-extraction step)
    joined with ``all_predictions.csv`` (track_id → predicted class) to
    give per-frame predicted positions. Cached by file path so the 100
    MB features csv is parsed once per session.

    Returns a DataFrame with columns ``TrackMate_Track_ID, t, z, y, x,
    class`` (class lowercased to line up with the per-class palette)."""
    feat = pd.read_csv(
        features_csv,
        usecols=["TrackMate_Track_ID", "t", "z", "y", "x"],
    )
    allp = pd.read_csv(all_predictions_csv)
    if "TrackMate_Track_ID" not in allp.columns or "Predicted_Class" not in allp.columns:
        log.warning(
            "all_predictions csv missing required cols (got %s)", list(allp.columns)
        )
        return pd.DataFrame(
            columns=["TrackMate_Track_ID", "t", "z", "y", "x", "class"]
        )
    joined = feat.merge(allp, on="TrackMate_Track_ID", how="inner")
    joined = joined.rename(columns={"Predicted_Class": "class"})
    joined["class"] = joined["class"].astype(str).str.lower()
    return joined[["TrackMate_Track_ID", "t", "z", "y", "x", "class"]]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("inception-demo")


# ---------------------------------------------------------------- paths
# kapoorlabslustre mounted locally via sshfs. The submit_inception_job.sh
# script SSHes to Jean Zay and resolves the lustre-side mirror of these.
JEANZAY_MOUNT = Path("/home/debian/jean-zay/demo_inception")
UPLOADS_DIR = JEANZAY_MOUNT / "uploads"
RESULTS_DIR = JEANZAY_MOUNT / "results"
MODELS_DIR = Path("/home/debian/jean-zay/models_inception")
USAGE_LOG = JEANZAY_MOUNT / "usage_log.jsonl"

# Lustre-side path the Jean Zay compute nodes see (used when we build
# CLI arguments for the SLURM job — never on the local side).
LUSTRE_DEMO = Path("/lustre/fsn1/projects/rech/jsy/uzj81mi/demo_inception")
LUSTRE_MODELS = Path("/lustre/fsn1/projects/rech/jsy/uzj81mi/models_inception")

SUBMIT_SCRIPT = Path(__file__).parent / "submit_inception_job.sh"


# ---------------------------------------------------------------- ORCID auth
ORCID_AUTH_URL = "https://orcid.org/oauth/authorize"
ORCID_TOKEN_URL = "https://orcid.org/oauth/token"
HEAVY_QUOTA_WINDOW_HOURS = 24
HEAVY_QUOTA_PER_WINDOW = 20


def orcid_oauth_config():
    """Return ``(client_id, client_secret, redirect_uri)`` or ``(None, None, None)``."""
    try:
        cfg = st.secrets["orcid"]
        return cfg["client_id"], cfg["client_secret"], cfg["redirect_uri"]
    except Exception:
        return None, None, None


def build_orcid_login_url(client_id: str, redirect_uri: str) -> str:
    params = {
        "client_id": client_id,
        "response_type": "code",
        "scope": "/authenticate",
        "redirect_uri": redirect_uri,
    }
    return f"{ORCID_AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_orcid_code(code, client_id, client_secret, redirect_uri):
    """Exchange the OAuth code for an access token + verified ORCID iD."""
    data = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        }
    ).encode()
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


def quota_status(orcid: str) -> tuple[int, datetime | None]:
    """Count this ORCID's submissions in the last 24h. Same windowed
    accounting as the oneat remote_app — submissions outside the window
    don't accumulate."""
    if not USAGE_LOG.exists() or not orcid:
        return 0, None
    cutoff = datetime.now(timezone.utc) - timedelta(hours=HEAVY_QUOTA_WINDOW_HOURS)
    count, oldest_ts = 0, None
    with USAGE_LOG.open() as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("orcid") != orcid or e.get("slurm_id") is None:
                continue
            try:
                ts = datetime.fromisoformat(e["ts"].replace("Z", "+00:00"))
            except Exception:
                continue
            if ts < cutoff:
                continue
            count += 1
            if oldest_ts is None or ts < oldest_ts:
                oldest_ts = ts
    return count, oldest_ts


def quota_embargo_until(orcid: str) -> datetime | None:
    count, oldest_ts = quota_status(orcid)
    if count >= HEAVY_QUOTA_PER_WINDOW and oldest_ts is not None:
        embargo_end = oldest_ts + timedelta(hours=HEAVY_QUOTA_WINDOW_HOURS)
        if embargo_end > datetime.now(timezone.utc):
            return embargo_end
    return None


def log_usage(event: dict) -> None:
    """Append one JSON record to the usage log. Best-effort — never blocks
    the user flow on a write failure."""
    event = dict(event)
    event.setdefault("ts", datetime.now(timezone.utc).isoformat())
    try:
        USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with USAGE_LOG.open("a") as fh:
            fh.write(json.dumps(event) + "\n")
    except Exception as e:
        log.warning("usage log write failed: %s", e)


# ---------------------------------------------------------------- demo + model discovery
def discover_demos() -> list[dict]:
    """A demo is a subdirectory of ``UPLOADS_DIR`` that has at least one
    TIFF timelapse + a ``ground_truth/`` subdir. Anything else is shown
    but flagged as incomplete."""
    if not UPLOADS_DIR.exists():
        return []
    out = []
    for entry in sorted(UPLOADS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        tifs = sorted(entry.glob("*.tif")) + sorted(entry.glob("*.tiff"))
        xmls = sorted(entry.glob("*.xml"))
        gt_dir = entry / "ground_truth"
        gt_csvs = sorted(gt_dir.glob("*.csv")) if gt_dir.is_dir() else []
        out.append(
            {
                "name": entry.name,
                "path": entry,
                "tifs": tifs,
                "xmls": xmls,
                "gt_csvs": gt_csvs,
                "complete": bool(tifs and gt_csvs),
            }
        )
    return out


def discover_models() -> list[Path]:
    """Each model is a directory under ``MODELS_DIR`` containing at least
    one ``.ckpt`` + a ``training_config.json``."""
    if not MODELS_DIR.exists():
        return []
    out = []
    for entry in sorted(MODELS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if not any(entry.rglob("*.ckpt")):
            continue
        out.append(entry)
    return out


def to_lustre_path(local_path: Path) -> str:
    """Translate a path on the sshfs mount to its Lustre equivalent so the
    Jean Zay compute node can see it."""
    local_str = str(local_path)
    if local_str.startswith(str(JEANZAY_MOUNT)):
        rel = Path(local_str).relative_to(JEANZAY_MOUNT)
        return str(LUSTRE_DEMO / rel)
    if local_str.startswith(str(MODELS_DIR)):
        rel = Path(local_str).relative_to(MODELS_DIR)
        return str(LUSTRE_MODELS / rel)
    return local_str  # already lustre-side


# ---------------------------------------------------------------- SLURM submit + poll
def submit_to_jeanzay(
    job_id: str,
    model_dir: Path,
    xml_path: Path,
    demo_name: str,
    tracklet_length: int,
    time_window_start: int,
    time_window_end: int,
) -> str | None:
    """Returns the SLURM job id (string) or None on failure."""
    cmd = [
        "bash",
        str(SUBMIT_SCRIPT),
        job_id,
        to_lustre_path(model_dir),
        to_lustre_path(xml_path),
        demo_name,
        str(tracklet_length),
        str(time_window_start),
        str(time_window_end),
    ]
    try:
        proc = subprocess.run(
            cmd, check=True, text=True, capture_output=True, timeout=60
        )
    except subprocess.CalledProcessError as e:
        st.error(f"sbatch failed: {e.stderr or e.stdout}")
        return None
    except subprocess.TimeoutExpired:
        st.error("sbatch timed out (60s). SSH to Jean Zay may be slow / down.")
        return None

    # sbatch prints "Submitted batch job <id>"
    for line in proc.stdout.splitlines():
        parts = line.strip().split()
        if parts and parts[-1].isdigit():
            return parts[-1]
    return None


def get_job_status(slurm_id: str) -> str:
    """Best-effort SLURM status. Returns one of PENDING / RUNNING /
    COMPLETED / FAILED / UNKNOWN."""
    try:
        proc = subprocess.run(
            [
                "ssh",
                "uzj81mi@jean-zay.idris.fr",
                "sacct",
                "-j",
                slurm_id,
                "-X",
                "--format=State",
                "--noheader",
                "--parsable2",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=20,
        )
    except Exception:
        return "UNKNOWN"
    state = proc.stdout.strip().split("\n")[0].strip()
    return state or "UNKNOWN"


def get_results(job_id: str) -> list[Path]:
    """Predicted per-class CSVs that ``predict-cellfate.py`` writes into
    ``RESULTS_DIR/<job_id>/`` once the run completes."""
    out_dir = RESULTS_DIR / job_id
    if not out_dir.is_dir():
        return []
    return sorted(out_dir.glob("*.csv"))


# ---------------------------------------------------------------- ground truth + predictions

# ---------------------------------------------------------------- predicted CSV reader
# predict-cellfate.py writes per-class files as
# ``<window_prefix>_<class>_predictions.csv`` (e.g.
# ``t0-191_basal_predictions.csv``), plus ``<prefix>_all_predictions.csv``
# (long-form all classes), plus ``<prefix>_confusion_matrix.csv``,
# ``<prefix>_transition_*.csv``. We want only the per-class files.
import re as _re

_NON_PER_CLASS_TOKENS = (
    "confusion_matrix",
    "transition",
    "_full",
    "all_predictions",   # long-form superset, not per-class
)

# Time-window prefix: ``t0-191_`` / ``t100-191_`` / ``t100p0-191_`` etc.
_WINDOW_PREFIX_RE = _re.compile(r"^t\d[\w\-]*?_")


def _extract_class_name(stem: str) -> str:
    """Pull a clean class name out of a predict-cellfate per-class
    filename:

        ``t0-191_basal_predictions``  → ``Basal``
        ``basal_predictions``         → ``Basal``
        ``Basal``                     → ``Basal``  (legacy)

    Falls back to the raw stem when nothing recognisable is there."""
    s = stem
    # Strip the trailing ``_predictions`` suffix the prediction script emits.
    if s.endswith("_predictions"):
        s = s[: -len("_predictions")]
    # Strip a leading ``t<digits>...`` time-window prefix if present.
    s = _WINDOW_PREFIX_RE.sub("", s, count=1)
    return s.capitalize() if s else stem


def load_class_csvs(csv_paths: list[Path]) -> pd.DataFrame:
    """Concatenate predict-cellfate.py's per-class output CSVs into one
    long DataFrame with a clean ``class`` column. Files that aren't
    per-class lists (confusion matrix, transition tables, the
    ``all_predictions`` long-form superset) are filtered out; rows
    without a recognised track-id column are dropped.

    predict-cellfate's per-class CSV ships ``TrackMate_Track_ID`` as the
    track key (we rename it to the canonical ``Track_ID`` so downstream
    display code only has to know one name).
    """
    keep_paths = [
        p for p in csv_paths
        if not any(tok in p.stem for tok in _NON_PER_CLASS_TOKENS)
    ]
    rows = []
    for p in keep_paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            log.warning("could not read %s: %s", p, e)
            continue
        for candidate in (
            "Track_ID",
            "TrackMate_Track_ID",
            "track_id",
            "trackmate_id",
            "TrackID",
        ):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "Track_ID"})
                break
        if "Track_ID" not in df.columns:
            log.warning(
                "skipping %s: no Track_ID column (got %s)",
                p, list(df.columns),
            )
            continue
        df["class"] = _extract_class_name(p.stem)
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["Track_ID", "class"])
    full = pd.concat(rows, ignore_index=True)
    full["Track_ID"] = pd.to_numeric(full["Track_ID"], errors="coerce")
    return full.dropna(subset=["Track_ID"])


# ---------------------------------------------------------------- lazy TIFF + overlay
class LazyTimelapse:
    """One-timepoint-at-a-time view over a TZYX (or TYX) TIFF.

    Uses :func:`tifffile.memmap` so the whole file is mapped lazily
    from disk and only the bytes for the requested timepoint actually
    get pulled in. Works for both single-page BigTIFFs (where the
    whole (T, Z, Y, X) volume is one giant page — our fifth_dataset
    case, ~9 GB uint8) and multi-page TIFFs.

    Falls back to ``series.asarray(key=t)`` page-indexing only when
    memmap fails (e.g. compressed TIFFs that can't be mapped).
    """

    def __init__(self, path: Path):
        from tifffile import memmap as _memmap

        self.path = Path(path)
        self._tf = TiffFile(self.path)
        series = self._tf.series[0]
        self.shape = tuple(series.shape)
        self.dtype = series.dtype
        self._series = series

        # Try memmap first — works for the single-page BigTIFF layout
        # the user's fifth_dataset uses (len(pages) == 1, with the
        # full (T, Z, Y, X) volume packed inside).
        try:
            self._memmap = _memmap(self.path)
        except Exception as e:
            log.warning(
                "tifffile.memmap failed for %s (%s); will fall back to "
                "series.asarray(key=t) page-indexing.",
                self.path, e,
            )
            self._memmap = None

    @property
    def n_timepoints(self) -> int:
        return int(self.shape[0]) if len(self.shape) >= 4 else 1

    def __getitem__(self, t: int) -> np.ndarray:
        nT = self.n_timepoints
        if t < 0 or t >= nT:
            raise IndexError(f"t={t} out of range (n_timepoints={nT})")
        if self._memmap is not None:
            # Slicing a memmap pulls only the bytes for that frame
            # from disk. ``np.asarray`` materialises it as an ndarray
            # so downstream code (max-Z projection) can work on it
            # without keeping the mmap mapping alive.
            if len(self.shape) >= 4:
                return np.asarray(self._memmap[t])
            return np.asarray(self._memmap)
        # Page-indexed fallback for layouts memmap can't handle.
        if len(self.shape) >= 4:
            return self._series.asarray(key=t)
        return self._series.asarray()

    def close(self):
        if self._memmap is not None:
            del self._memmap
            self._memmap = None
        self._tf.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()




# ---------------------------------------------------------------- main
def main():
    st.set_page_config(page_title="Inception Cell-Fate Demo", layout="wide")
    st.title("Inception Cell-Fate Prediction")
    st.markdown(
        "Run a trained Inception cell-fate classifier on a TrackMate XML, "
        "compare per-track predictions to ground-truth CSVs, and inspect "
        "the timelapse overlay."
    )

    # ── ORCID auth gate (same shape / placement as the oneat app) ────
    st.sidebar.header("User (ORCID)")
    client_id, client_secret, redirect_uri = orcid_oauth_config()

    # Handle the OAuth callback: ORCID redirects back with ?code=...
    qp = st.query_params
    code = qp.get("code")
    if code and not st.session_state.get("verified_orcid") and client_id:
        with st.spinner("Verifying ORCID sign-in..."):
            tok = exchange_orcid_code(
                code, client_id, client_secret, redirect_uri
            )
        if tok and tok.get("orcid"):
            st.session_state["verified_orcid"] = tok["orcid"]
            st.session_state["verified_name"] = tok.get("name")
            log.info("ORCID OAuth success for %s", tok["orcid"])
        else:
            st.sidebar.error("ORCID sign-in failed. Please try again.")
        # Clear ?code from URL so a refresh doesn't re-exchange it.
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
            f"Signed in as {verified_name or verified_orcid}\n\n"
            f"`{verified_orcid}`"
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

    # Quota / embargo accounting, surfaced as small captions like in the
    # oneat sidebar so the user can see their remaining slot at a glance.
    embargo_end = (
        quota_embargo_until(verified_orcid) if verified_orcid else None
    )
    runs_window = quota_status(verified_orcid)[0] if verified_orcid else 0
    can_submit_quota = bool(verified_orcid) and embargo_end is None
    if verified_orcid:
        st.sidebar.caption(
            f"Runs in last {HEAVY_QUOTA_WINDOW_HOURS}h: "
            f"{runs_window} / {HEAVY_QUOTA_PER_WINDOW}"
        )
        if embargo_end is not None:
            st.sidebar.warning(
                f"Quota exceeded. Embargo until "
                f"{embargo_end.strftime('%Y-%m-%d %H:%M UTC')}."
            )
        else:
            st.sidebar.success("Prediction submission enabled.")

    # The rest of the page uses ``orcid`` as the auth flag — keep the
    # name so the existing downstream guards (Run button etc.) don't
    # need to change.
    orcid = verified_orcid

    # ── demo + model pickers ───────────────────────────────────────
    demos = discover_demos()
    models = discover_models()

    if not demos:
        st.warning(
            f"No demo subdirs found under {UPLOADS_DIR}. "
            "Mount the Lustre sshfs share + ensure each demo has at "
            "least one TIFF and a ground_truth/ folder."
        )
    if not models:
        st.warning(
            f"No model folders found under {MODELS_DIR}. "
            "Each model must be a directory with at least one .ckpt + "
            "training_config.json."
        )

    st.sidebar.header("Demo")
    demo_names = [d["name"] for d in demos] or ["(none)"]
    demo_name = st.sidebar.selectbox("Demo dataset", demo_names)
    demo = next((d for d in demos if d["name"] == demo_name), None)

    st.sidebar.header("Model")
    model_names = [m.name for m in models] or ["(none)"]
    # Default to the ``*_medium`` checkpoint when present — the
    # alphabetical sort otherwise picks ``*_high`` first.
    default_idx = next(
        (i for i, n in enumerate(model_names) if "medium" in n.lower()),
        0,
    )
    model_name = st.sidebar.selectbox(
        "Checkpoint folder", model_names, index=default_idx
    )
    model_dir = next((m for m in models if m.name == model_name), None)

    # ── XML source ────────────────────────────────────────────────
    # Demos are curated — uploads are disabled. Each demo dir is
    # expected to ship its own ``.xml`` (typically one). The first one
    # is the default; the dropdown only appears when a demo has more
    # than one XML staged.
    st.sidebar.header("TrackMate XML")
    xml_path = None
    if demo and demo["xmls"]:
        if len(demo["xmls"]) == 1:
            xml_path = demo["xmls"][0]
            st.sidebar.text(f"XML: {xml_path.name}")
        else:
            xml_pick = st.sidebar.selectbox(
                "Pick XML",
                [p.name for p in demo["xmls"]],
            )
            xml_path = demo["path"] / xml_pick
    elif demo:
        st.sidebar.warning(
            "No .xml found in this demo dir. "
            "Drop one into "
            f"{demo['path'].relative_to(JEANZAY_MOUNT.parent)}."
        )

    # ── prediction knobs ──────────────────────────────────────────
    st.sidebar.header("Prediction parameters")

    # Peek at the demo's TIFF header (TZYX) to surface the actual T
    # count to the user — so when they pick a "start frame" they can
    # see whether 100 is plausibly mid-movie or already past the end.
    # Header-only read: no pixel data is touched.
    n_timepoints = None
    if demo and demo["tifs"]:
        try:
            with TiffFile(demo["tifs"][0]) as tf:
                shape = tuple(tf.series[0].shape)
                n_timepoints = (
                    int(shape[0]) if len(shape) >= 4 else 1
                )
        except Exception as e:
            log.warning(
                "could not read TIFF header for %s: %s",
                demo["tifs"][0], e,
            )

    if n_timepoints is not None:
        st.sidebar.info(
            f"This demo's TIFF has **{n_timepoints} timepoints** "
            f"(valid frame indices: 0 … {n_timepoints - 1}). "
            f"Use the window inputs below to focus prediction on a "
            f"subrange — e.g. start={max(0, n_timepoints - 100)}, "
            f"end=-1 covers the last 100 frames."
        )
        # Biological context — the developing system hasn't committed to a
        # fate at early timepoints, so the classifier can't either.
        # Accuracy climbs sharply once you start the window past frame
        # ~100 in this dataset; surface that as a warning so users don't
        # come away thinking the model is broken.
        st.sidebar.warning(
            "**About time-window choice.**  This is a *developing* "
            "system — at early timepoints the cells haven't committed "
            "to a fate yet, so the classifier has nothing to latch onto "
            "and accuracy is low. Starting the window at frame **100 or "
            "later** captures cells whose fates are already determined, "
            "and prediction accuracy jumps dramatically."
        )

    tracklet_length = st.sidebar.slider(
        "Time-window minimum duration (frames)",
        min_value=25,
        max_value=200,
        value=25,
        step=1,
        help=(
            "Tracks shorter than this are ignored. Equivalent to "
            "``parameters.tracklet_length`` in predict-cellfate.py."
        ),
    )
    tw_start_max = (n_timepoints - 1) if n_timepoints is not None else 10_000
    tw_start = st.sidebar.number_input(
        "Time-window start (frame)",
        min_value=0,
        max_value=tw_start_max,
        value=0,
        step=1,
        help=(
            f"Frame index where prediction begins. "
            f"This dataset has {n_timepoints} frames, so values up to "
            f"{tw_start_max} are valid."
            if n_timepoints is not None
            else "Frame index where prediction begins."
        ),
    )
    tw_end = st.sidebar.number_input(
        "Time-window end (frame, -1 = last)",
        min_value=-1,
        max_value=tw_start_max,
        value=-1,
        step=1,
        help=(
            f"Frame index where prediction ends. -1 = last available "
            f"frame ({n_timepoints - 1 if n_timepoints else 'N-1'}). "
            f"Must be > start, and the span (end - start + 1) must be "
            f"larger than the tracklet length above."
            if n_timepoints is not None
            else "Frame index where prediction ends. -1 = last."
        ),
    )

    # ── submit ────────────────────────────────────────────────────
    # Quota + embargo state already computed in the ORCID gate above
    # (``can_submit_quota``). Just gate on the remaining inputs here.
    can_submit = bool(
        orcid and demo and model_dir and xml_path and can_submit_quota
    )

    st.header("Submit prediction job")
    if st.button(
        "Run on Jean Zay", type="primary", disabled=not can_submit
    ):
        job_id = uuid.uuid4().hex[:12]
        slurm_id = submit_to_jeanzay(
            job_id=job_id,
            model_dir=model_dir,
            xml_path=xml_path,
            demo_name=demo["name"],
            tracklet_length=int(tracklet_length),
            time_window_start=int(tw_start),
            time_window_end=int(tw_end),
        )
        if slurm_id:
            st.session_state["pending_job"] = {
                "job_id": job_id,
                "slurm_id": slurm_id,
                "demo": demo["name"],
                "model": model_dir.name,
                "tracklet_length": int(tracklet_length),
            }
            log_usage(
                {
                    "orcid": orcid,
                    "demo": demo["name"],
                    "model": model_dir.name,
                    "tracklet_length": int(tracklet_length),
                    "slurm_id": slurm_id,
                    "job_id": job_id,
                }
            )
            st.success(
                f"Submitted SLURM job {slurm_id} (internal id {job_id}). "
                "Polling for completion below."
            )
        else:
            st.error("sbatch did not return a job id.")

    # ── job status polling ────────────────────────────────────────
    pending = st.session_state.get("pending_job")
    if pending:
        state = get_job_status(pending["slurm_id"])
        # Bucket sacct states into a short user-facing category +
        # emoji, same shape as the oneat app's refresh-status badge.
        _SLURM_BUCKETS = {
            "PENDING":   ("queued",    "🟡"),
            "CONFIGURING": ("queued",  "🟡"),
            "RUNNING":   ("running",   "🔵"),
            "COMPLETING": ("running",  "🔵"),
            "COMPLETED": ("completed", "🟢"),
            "FAILED":    ("crashed",   "🔴"),
            "TIMEOUT":   ("crashed",   "🔴"),
            "CANCELLED": ("crashed",   "🔴"),
            "OUT_OF_MEMORY": ("crashed", "🔴"),
            "NODE_FAIL": ("crashed",   "🔴"),
            "PREEMPTED": ("crashed",   "🔴"),
            "BOOT_FAIL": ("crashed",   "🔴"),
            "UNKNOWN":   ("unknown",   "⚪"),
        }
        bucket, emoji = _SLURM_BUCKETS.get(state.split()[0] if state else "UNKNOWN", ("unknown", "⚪"))

        col_state, col_btn = st.columns([3, 1])
        with col_state:
            st.markdown(
                f"### {emoji} **{bucket.upper()}** — `{state}`"
            )
            st.caption(
                f"SLURM job {pending['slurm_id']}  ·  "
                f"internal id {pending['job_id']}"
            )
        with col_btn:
            if st.button("Refresh now", use_container_width=True):
                st.rerun()

        if bucket == "completed":
            csvs = get_results(pending["job_id"])
            if csvs:
                st.session_state["last_results"] = {
                    **pending,
                    "result_csvs": csvs,
                }
                st.session_state.pop("pending_job", None)
                st.success(f"Results ready: {len(csvs)} CSV(s).")
                st.rerun()
            else:
                st.warning(
                    "Job marked complete but no CSVs landed in "
                    f"{RESULTS_DIR / pending['job_id']}."
                )
        elif bucket == "crashed":
            st.error(
                f"Job failed (SLURM `{state}`). Check the SLURM log under "
                f"{RESULTS_DIR / pending['job_id']}/."
            )
        else:
            # Still queued / running — keep streamlit's auto-poll loop,
            # the SAME way the oneat app does it: ``time.sleep + rerun``.
            # An earlier version used a browser-side <meta refresh>
            # because it stays interactive during the wait — but a full
            # browser reload drops streamlit's WebSocket session,
            # which wipes ``verified_orcid`` and forces the user to
            # sign in again every 2 minutes. ``st.rerun()`` keeps the
            # session state alive.
            #
            # Trade-off: the "Refresh now" button can only fire AFTER
            # the sleep completes (the streamlit thread is parked in
            # ``time.sleep``). Sleeping in 5-second slices instead of
            # one 120s block keeps the button responsive within 5s
            # while still polling sacct roughly every 2 minutes.
            st.info(
                "Auto-refreshing every 2 minutes. "
                "Click *Refresh now* for an immediate poll."
            )
            POLL_TOTAL_S = 120
            POLL_SLICE_S = 5
            for _ in range(POLL_TOTAL_S // POLL_SLICE_S):
                time.sleep(POLL_SLICE_S)
            st.rerun()

    # ── results: read predict-cellfate.py outputs verbatim ───────
    last = st.session_state.get("last_results")
    if last and demo:
        st.markdown("---")
        st.header(f"Results · {last['demo']} · {last['model']}")

        # All metrics + GT matching happen on Jean Zay. The local box
        # only reads the tiny output files off the sshfs mount.
        # predict-cellfate.py prefixes every file it writes with the
        # time-window tag (e.g. ``t0-191_confusion_matrix.csv``,
        # ``t0-191_basal_predictions.csv``), so we glob the suffix.
        result_dir = RESULTS_DIR / last["job_id"]
        cm_csv_matches = sorted(result_dir.glob("*confusion_matrix.csv"))
        cm_png_matches = sorted(result_dir.glob("*confusion_matrix.png"))

        tab_cm, tab_pred, tab_viewer = st.tabs(
            ["Confusion matrix", "Predicted track IDs", "Timelapse viewer"]
        )

        with tab_cm:
            # predict-cellfate.py emits one ``t<start>-<end>_confusion_matrix.csv``
            # per block (plus a global one over the full time window).
            # Group csv/png pairs by their ``t<a>-<b>`` tag and expose a
            # numeric-sorted slider when there's more than one block.
            import re as _re_cm

            _WINDOW_TAG_RE = _re_cm.compile(r"t(?P<a>-?\d+)-(?P<b>-?\d+)")

            def _window_key(p: Path) -> str:
                stem = p.stem
                if stem.endswith("_confusion_matrix"):
                    stem = stem[: -len("_confusion_matrix")]
                return stem or "(global)"

            def _window_sort_key(label: str) -> tuple[int, int]:
                m = _WINDOW_TAG_RE.search(label)
                if m is None:
                    return (10**9, 10**9)  # sort unparseable to the end
                return (int(m.group("a")), int(m.group("b")))

            blocks: dict[str, dict[str, Path]] = {}
            for p in cm_csv_matches:
                blocks.setdefault(_window_key(p), {})["csv"] = p
            for p in cm_png_matches:
                blocks.setdefault(_window_key(p), {})["png"] = p

            if not blocks:
                st.info(
                    "No confusion matrix in this results dir — "
                    "predict-cellfate.py only writes one when GT CSVs "
                    "are configured (check the SLURM log for the run; "
                    "missing GT paths print `No GT annotation files "
                    "found; skipping confusion matrix.`)."
                )
            else:
                labels = sorted(blocks.keys(), key=_window_sort_key)
                if len(labels) == 1:
                    pick = labels[0]
                else:
                    pick = st.select_slider(
                        "Time block (start → end frame)",
                        options=labels,
                        value=labels[0],
                        key="cm_block_pick",
                    )
                pair = blocks[pick]
                cm_png = pair.get("png")
                cm_csv = pair.get("csv")
                if cm_png is not None and cm_png.is_file():
                    st.image(
                        str(cm_png),
                        caption=cm_png.name,
                        use_container_width=True,
                    )
                if cm_csv is not None and cm_csv.is_file():
                    cm_df = pd.read_csv(cm_csv, index_col=0)
                    st.caption(cm_csv.name)
                    st.dataframe(cm_df, use_container_width=True)
                    total = int(cm_df.values.sum())
                    if total:
                        correct = int(np.trace(cm_df.values))
                        st.metric(
                            "Accuracy (GT tracks correctly predicted)",
                            f"{correct}/{total} = {correct/total:.3f}",
                        )

        with tab_pred:
            feat_csvs = sorted(demo["path"].glob("features_*.csv"))
            all_pred_csvs = sorted(result_dir.glob("*all_predictions.csv"))

            if not feat_csvs:
                st.info(
                    "No `features_*.csv` next to the demo XML — "
                    "predict-cellfate.py writes it during the first "
                    "XML-mode run. Re-run the job."
                )
            elif not all_pred_csvs:
                st.info(
                    "No `all_predictions.csv` in the result dir — "
                    "the prediction job hasn't finished."
                )
            else:
                # Per-track aggregated feature df (one row per
                # TrackMate_Track_ID, mean over all frames).
                agg = _load_features_aggregated(str(feat_csvs[0]))

                # Predicted class per track (long-form table).
                all_pred = pd.read_csv(all_pred_csvs[0])
                if not {"TrackMate_Track_ID", "Predicted_Class"}.issubset(
                    all_pred.columns
                ):
                    st.error(
                        f"all_predictions.csv has unexpected columns "
                        f"({list(all_pred.columns)})."
                    )
                    st.stop()

                # GT class per track — prefer the server-side dump
                # (matching done inside predict-cellfate.py with the
                # exact same logic as the confusion matrix); fall back
                # to local nearest-neighbour matching when missing.
                gt_assign_csvs = sorted(
                    result_dir.glob("*gt_track_assignments.csv")
                )
                if gt_assign_csvs:
                    gt_assign = pd.read_csv(gt_assign_csvs[0])
                    gt_source = f"server-side ({gt_assign_csvs[0].name})"
                elif demo["gt_csvs"]:
                    gt_assign = _match_gt_to_tracks(
                        str(feat_csvs[0]),
                        tuple(str(p) for p in demo["gt_csvs"]),
                    )
                    gt_source = (
                        "local fallback (nearest-neighbour matching on "
                        "this box — re-run predict-cellfate.py to get "
                        "the authoritative server-side assignment)"
                    )
                else:
                    gt_assign = pd.DataFrame(
                        columns=["TrackMate_Track_ID", "GT_Class"]
                    )
                    gt_source = "no GT CSVs"

                # Join the per-track features with both labels.
                gt_panel = agg.merge(gt_assign, on="TrackMate_Track_ID", how="inner")
                pred_panel = agg.merge(
                    all_pred[["TrackMate_Track_ID", "Predicted_Class"]],
                    on="TrackMate_Track_ID",
                    how="inner",
                )

                st.caption(
                    f"GT tracks: {len(gt_panel)} · Predicted tracks: "
                    f"{len(pred_panel)} · GT source: {gt_source}"
                )

                # Restrict the selector to features actually present in
                # this CSV. Default to the curated picks.
                available = [c for c in FEATURE_COLUMNS if c in agg.columns]
                if not available:
                    st.warning(
                        "None of the expected shape/motion feature "
                        "columns are in the features CSV — got "
                        f"{list(agg.columns)}."
                    )
                else:
                    defaults = [
                        c for c in DEFAULT_FEATURE_PICKS if c in available
                    ] or available[:4]
                    picks = st.multiselect(
                        "Features (GT left · Predicted right)",
                        options=available,
                        default=defaults,
                        key="feat_pick",
                    )
                    for feat in picks:
                        col_l, col_r = st.columns(2)
                        with col_l:
                            fig = _violin_by_class(
                                gt_panel, feat, "GT_Class",
                                title=f"GT · {feat}",
                            )
                            st.pyplot(fig)
                            plt.close(fig)
                        with col_r:
                            fig = _violin_by_class(
                                pred_panel, feat, "Predicted_Class",
                                title=f"Pred · {feat}",
                            )
                            st.pyplot(fig)
                            plt.close(fig)

            # Downloads — every prediction CSV the job emitted.
            st.divider()
            st.markdown("**Download prediction CSVs**")
            dl_csvs = sorted(
                p for p in result_dir.glob("*.csv")
                if "transition" not in p.stem
                and "confusion_matrix" not in p.stem
                and "gt_track_assignments" not in p.stem
            )
            if not dl_csvs:
                st.info("No prediction CSVs in the result dir.")
            for p in dl_csvs:
                st.download_button(
                    label=f"⤓ {p.name}",
                    data=p.read_bytes(),
                    file_name=p.name,
                    mime="text/csv",
                    key=f"dl_{p.name}",
                )

        with tab_viewer:
            if not demo["tifs"]:
                st.info("No TIFF timelapse in this demo dir.")
            else:
                tif_pick = st.selectbox(
                    "TIFF",
                    [p.name for p in demo["tifs"]],
                    key="tif_pick",
                )
                tif_path = demo["path"] / tif_pick

                with LazyTimelapse(tif_path) as tl:
                    nT = tl.n_timepoints
                    t = st.slider(
                        "Time point",
                        min_value=0,
                        max_value=max(0, nT - 1),
                        value=max(0, nT - 1),
                        key="viewer_t",
                    )
                    raw_frame = tl[t]
                st.caption(
                    f"t = {t} of {nT - 1}. Coords are pixel-native — "
                    "GT and per-frame predictions are plotted as-is, no "
                    "calibration division (the CSV ranges already fit "
                    "the TIFF extent)."
                )

                if raw_frame.ndim == 3:
                    max_proj = np.max(raw_frame, axis=0)
                else:
                    max_proj = raw_frame
                vmin, vmax = np.percentile(max_proj, (1, 99.8))
                if vmax > vmin:
                    img = np.clip(
                        (max_proj - vmin) / (vmax - vmin + 1e-9), 0, 1
                    )
                else:
                    img = np.zeros_like(max_proj, dtype=np.float32)

                # GT: one csv per class in demo['ground_truth']/, sparse
                # (only annotated frames). Read as-is, filter to t==picked.
                gt_frames = []
                for gt_csv in demo["gt_csvs"]:
                    cls = _GT_FILENAME_TO_CLASS.get(
                        gt_csv.name, gt_csv.stem.split("_")[0]
                    )
                    pos = _load_positions(gt_csv, cls)
                    if pos is not None:
                        gt_frames.append(pos)
                gt_df = (
                    pd.concat(gt_frames, ignore_index=True)
                    if gt_frames
                    else pd.DataFrame(columns=["t", "z", "y", "x", "class"])
                )

                # Predictions: per-class csvs in the result dir only
                # carry each track's FIRST timepoint (one row per
                # track), so filtering them by t gives 0 cells on most
                # frames. Pull per-frame trajectories from the demo's
                # features csv and join with all_predictions.csv to
                # tag each row with its predicted class.
                feat_csvs = sorted(demo["path"].glob("features_*.csv"))
                all_pred_csvs = sorted(
                    result_dir.glob("*all_predictions.csv")
                )
                pred_df_pos = pd.DataFrame(
                    columns=["t", "z", "y", "x", "class"]
                )
                if feat_csvs and all_pred_csvs:
                    pred_df_pos = _load_per_frame_predictions(
                        str(feat_csvs[0]), str(all_pred_csvs[0])
                    )
                elif not feat_csvs:
                    st.info(
                        "No `features_*.csv` in the demo dir — per-frame "
                        "predictions can't be reconstructed. Re-run the "
                        "prediction job; the feature cache will be "
                        "written next to the XML."
                    )
                elif not all_pred_csvs:
                    st.info(
                        "No `all_predictions.csv` in the result dir — "
                        "the prediction job didn't finish."
                    )

                gt_here = gt_df[gt_df["t"].round().astype(int) == t]
                pred_here = pred_df_pos[
                    pred_df_pos["t"].round().astype(int) == t
                ]

                show_gt = st.checkbox(
                    "Show ground-truth cells", value=True, key="show_gt"
                )
                show_pred = st.checkbox(
                    "Show predicted cells", value=True, key="show_pred"
                )

                fig, ax = plt.subplots(figsize=(9, 9))
                ax.imshow(img, cmap="gray")
                drawn_classes = set()
                if show_gt and len(gt_here):
                    for cls in sorted(gt_here["class"].unique()):
                        rows = gt_here[gt_here["class"] == cls]
                        ax.scatter(
                            rows["x"], rows["y"],
                            s=80,
                            facecolors="none",
                            edgecolors=_class_color(cls),
                            linewidths=1.5,
                            label=f"GT {cls}",
                        )
                        drawn_classes.add(("gt", cls))
                if show_pred and len(pred_here):
                    for cls in sorted(pred_here["class"].unique()):
                        rows = pred_here[pred_here["class"] == cls]
                        ax.scatter(
                            rows["x"], rows["y"],
                            s=45,
                            marker="x",
                            c=_class_color(cls),
                            linewidths=1.8,
                            label=f"Pred {cls}",
                        )
                        drawn_classes.add(("pred", cls))
                if drawn_classes:
                    ax.legend(
                        loc="upper right",
                        framealpha=0.6,
                        fontsize=8,
                    )
                ax.set_title(
                    f"Max-Z   t={t}   "
                    f"GT={len(gt_here)} cells, Pred={len(pred_here)} cells",
                    fontsize=10,
                )
                ax.axis("off")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)


if __name__ == "__main__":
    main()

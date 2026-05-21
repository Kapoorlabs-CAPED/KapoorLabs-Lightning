"""Inception Cell-Fate napari plugin.

Runs the same Inception cell-fate model that the streamlit
``apps/streamlit/inception/remote_app.py`` and CLI
``scripts/model_prediction/predict-cellfate.py`` use — but directly
inside napari, without SLURM, against the timelapse currently in the
viewer. Per-track-id predictions are added as one Points layer per
class so the user can see which cells the model called Basal / Radial
/ Goblet / … overlaid on their data.

Optional: drop a directory of ground-truth CSVs into the GT field and
the plugin resolves each annotation row to its XML Track_ID
(NapaTrackMater pattern — divide physical coords by the XML's own
calibration to get pixel space, then nearest-neighbour KDTree match),
joins predictions to GT on Track_ID, and renders a confusion matrix +
per-class accuracy table.

The user supplies the model checkpoint folder. A future revision will
auto-download pretrained checkpoints from HuggingFace; for now this
plugin expects a local model directory containing the ``.ckpt`` and
``training_config.json`` written by ``CareInception`` / similar
trainers.

Made by KapoorLabs, 2026.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
from magicgui import magicgui
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


# Shared class → hex palette. Same one the streamlit app uses.
CLASS_COLORS = (
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
)


def _abs_resource(relpath: str) -> str:
    """Plugin-local resource path (logo etc.)."""
    here = Path(__file__).resolve().parent
    return str((here / relpath).absolute())


def _class_color_map(class_names: List[str]) -> Dict[str, str]:
    """Deterministic class → hex color. Sorting input keeps the colour
    stable across reruns even when the present set of classes shifts."""
    return {
        c: CLASS_COLORS[i % len(CLASS_COLORS)]
        for i, c in enumerate(sorted(set(class_names)))
    }


# ─────────────────────────────────────────────── model loading + prediction
def _build_inception_network(arch: dict):
    """Build the right ``InceptionNet`` / ``DenseNet`` / ``MitosisNet``
    from the ``parameters`` block of a ``training_config.json``.

    The model_choice + per-arch knobs mirror exactly what
    ``predict-cellfate.py::build_network`` does — pulled out here so the
    plugin doesn't need to go through Hydra to instantiate the same
    network shape.
    """
    from kapoorlabs_lightning.pytorch_models import (
        DenseNet,
        InceptionNet,
        MitosisNet,
    )
    from kapoorlabs_lightning.tracking.track_features import (
        DYNAMIC_FEATURES,
        SHAPE_DYNAMIC_FEATURES,
        SHAPE_FEATURES,
    )

    model_choice = arch.get("model_choice", "inception")
    num_classes = int(arch.get("num_classes", 3))
    growth_rate = int(arch.get("growth_rate", 4))
    block_config = tuple(arch.get("block_config", [6]))
    num_init_features = int(arch.get("num_init_features", 32))
    bottleneck_size = int(arch.get("bottleneck_size", 4))
    kernel_size = int(arch.get("kernel_size", 7))
    attn_heads = int(arch.get("attn_heads", 8))
    seq_len = int(arch.get("seq_len", 25))

    features = arch.get("features", "shape_dynamic")
    if features == "shape":
        input_channels = len(SHAPE_FEATURES)
    elif features == "dynamic":
        input_channels = len(DYNAMIC_FEATURES)
    else:
        input_channels = len(SHAPE_DYNAMIC_FEATURES)

    if model_choice == "inception":
        return InceptionNet(
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
    if model_choice == "densenet":
        return DenseNet(
            input_channels=input_channels,
            num_classes=num_classes,
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=num_init_features,
            bottleneck_size=bottleneck_size,
            kernel_size=kernel_size,
        )
    return MitosisNet(
        input_channels=input_channels,
        num_classes=num_classes,
    )


def _load_inception_model(model_dir: Path) -> Tuple[object, dict]:
    """Load ``CellFateModule`` from a directory holding the ``.ckpt`` +
    ``training_config.json``.

    Returns ``(module, arch_dict)``. The arch dict is whatever the
    training run wrote into ``parameters`` so the caller has access to
    things like ``class_map``, ``features``, ``seq_len`` etc.
    """
    import torch

    from kapoorlabs_lightning.cellfate_module import CellFateModule
    from kapoorlabs_lightning.utils import load_checkpoint_model

    model_dir = Path(model_dir)
    cfg_path = model_dir / "training_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"training_config.json not found in {model_dir}. "
            "Plugin needs the JSON sidecar the trainer writes next to "
            "the .ckpt — without it we can't know which network shape "
            "to build before loading the weights."
        )
    cfg = json.loads(cfg_path.read_text())
    arch = cfg.get("parameters", cfg)
    network = _build_inception_network(arch)

    ckpt = load_checkpoint_model(str(model_dir))
    if ckpt is None:
        raise FileNotFoundError(f"No .ckpt found under {model_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = CellFateModule.load_from_checkpoint(
        ckpt,
        map_location=device,
        network=network,
        weights_only=False,
        strict=False,
    )
    module.eval()
    module.to(device)
    return module, arch


def _predict_cellfate(
    xml_path: Path,
    module: object,
    arch: dict,
    tracklet_length: int,
    time_window: Optional[Tuple[int, int]] = None,
) -> Tuple[pd.DataFrame, Dict[int, str], dict]:
    """Run the Inception prediction end-to-end.

    Reads features off the master XML (NapaTrackMater-style — same
    extractor ``predict-cellfate.py`` uses) and calls
    :func:`predict_all_tracks` to emit ``{Track_ID: class_name}``.
    Returns ``(features_df, predictions, class_map)``.
    """
    import torch

    from kapoorlabs_lightning.tracking.track_features import (
        DYNAMIC_FEATURES,
        SHAPE_DYNAMIC_FEATURES,
        SHAPE_FEATURES,
    )
    from kapoorlabs_lightning.tracking.track_prediction import (
        predict_all_tracks,
    )
    from kapoorlabs_lightning.tracking.track_vectors import TrackVectors

    # Build the features DataFrame the same way predict-cellfate.py does
    # for XML input mode. We always go through ``TrackVectors`` because
    # the streamlit demo + the slurm job both pass XML, not a feature CSV.
    tv = TrackVectors(str(xml_path))
    df = tv.dataframe.copy()
    # ``predict_all_tracks`` wants a ``t`` column (lowercase). The
    # TrackVectors dataframe already uses that convention.

    feature_choice = arch.get("features", "shape_dynamic")
    if feature_choice == "shape":
        feature_cols = [c for c in SHAPE_FEATURES if c in df.columns]
    elif feature_choice == "dynamic":
        feature_cols = [c for c in DYNAMIC_FEATURES if c in df.columns]
    else:
        feature_cols = [c for c in SHAPE_DYNAMIC_FEATURES if c in df.columns]

    class_map_raw = arch.get(
        "class_map", {0: "Basal", 1: "Radial", 2: "Goblet"}
    )
    class_map = {int(k): str(v) for k, v in dict(class_map_raw).items()}

    # Restrict to the user-chosen window, matching predict-cellfate.py's
    # ``time_window`` semantics.
    if time_window is not None:
        t_start, t_end = int(time_window[0]), int(time_window[1])
        if t_end == -1:
            t_end = int(df["t"].max())
        df = df[(df["t"] >= t_start) & (df["t"] <= t_end)].reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions = predict_all_tracks(
        dataframe=df,
        tracklet_length=int(tracklet_length),
        class_map=class_map,
        model=module.network,
        device=device,
        feature_columns=feature_cols,
    )
    return df, predictions, class_map


# ─────────────────────────────────────────────── GT resolution
def _resolve_gt_track_ids(
    xml_path: Path,
    gt_csv_paths: List[Path],
    space_veto: float = 5.0,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """Same shape as the streamlit app's ``resolve_gt_track_ids``.

    For each GT row ``(T, Z, Y, X)`` in physical units → divide by the
    XML's own calibration → KDTree-match to the closest spot at that
    frame → look up the host Track_ID. Track IDs that don't actually
    exist in the XML are dropped, so the accuracy report is computed
    only over GT entries the model could plausibly have predicted on.
    """
    from scipy.spatial import cKDTree

    from kapoorlabs_lightning.tracking.xml_parser import TrackMateXML

    tm = TrackMateXML(str(xml_path))
    cal = tm.calibration

    spot_to_track: Dict[int, int] = {}
    for track_id in tm.tracks:
        for sid in tm.get_all_spot_ids_for_track(track_id):
            spot_to_track[sid] = track_id

    spot_tuples, spot_ids = [], []
    for sid, spot in tm.spots.items():
        if sid not in spot_to_track:
            continue
        spot_tuples.append(
            (spot.frame, spot.z / cal.z, spot.y / cal.y, spot.x / cal.x)
        )
        spot_ids.append(sid)
    if not spot_tuples:
        return (
            pd.DataFrame(columns=["Track_ID", "class", "frame", "z", "y", "x"]),
            {"overall": {"total": 0, "matched": 0}},
        )

    arr = np.asarray(spot_tuples, dtype=np.float64)
    tree = cKDTree(arr)

    rows, stats = [], {}
    for csv_path in gt_csv_paths:
        try:
            gt = pd.read_csv(csv_path)
        except Exception:
            continue
        for src, dst in (
            ("FRAME", "T"),
            ("POSITION_X", "X"),
            ("POSITION_Y", "Y"),
            ("POSITION_Z", "Z"),
        ):
            if src in gt.columns and dst not in gt.columns:
                gt = gt.rename(columns={src: dst})
        if not all(c in gt.columns for c in ("T", "Z", "Y", "X")):
            continue
        gt_arr = np.stack(
            [
                gt["T"].astype(float).values,
                gt["Z"].astype(float).values / cal.z,
                gt["Y"].astype(float).values / cal.y,
                gt["X"].astype(float).values / cal.x,
            ],
            axis=1,
        )

        cls = csv_path.stem
        matched = 0
        for g in gt_arr:
            hits = tree.query_ball_point(g, space_veto, p=2)
            if not hits:
                continue
            d, idx = tree.query(g, k=1)
            if idx not in hits:
                continue
            if abs(arr[idx, 0] - g[0]) > 0:
                continue
            track_id = spot_to_track.get(spot_ids[idx])
            if track_id is None:
                continue
            rows.append(
                {
                    "Track_ID": int(track_id),
                    "class": cls,
                    "frame": int(g[0]),
                    "z": float(g[1]),
                    "y": float(g[2]),
                    "x": float(g[3]),
                }
            )
            matched += 1
        stats[cls] = {"total": int(len(gt_arr)), "matched": matched}

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(subset=["Track_ID"], keep="first").reset_index(
            drop=True
        )
    stats["overall"] = {
        "total": sum(s["total"] for s in stats.values() if "total" in s),
        "matched": sum(s["matched"] for s in stats.values() if "matched" in s),
    }
    return out, stats


def _build_prediction_points(
    df: pd.DataFrame,
    predictions: Dict[int, str],
    xml_path: Path,
    track_id_column: str = "Track_ID",
) -> Dict[str, np.ndarray]:
    """Group every spot in the features DataFrame by its predicted
    class, returning ``{class_name: (N, 4) array of (t, z_pix, y_pix,
    x_pix)}`` suitable for ``viewer.add_points``.

    The features DataFrame already has ``t, z, y, x`` in **pixel
    units** — ``TrackVectors`` builds it that way. So no conversion is
    needed here; we just split rows by predicted class.
    """
    if df.empty or not predictions:
        return {}
    # The track-id column in the features DataFrame is ``TrackMate_Track_ID``
    # (the parent track key) rather than ``Track_ID``. Pick whichever
    # column exists.
    for cand in ("TrackMate_Track_ID", "Track_ID", track_id_column):
        if cand in df.columns:
            track_col = cand
            break
    else:
        return {}

    pred_series = df[track_col].map(predictions)
    out: Dict[str, np.ndarray] = {}
    for cls in sorted(set(predictions.values())):
        rows = df[pred_series == cls]
        if rows.empty:
            continue
        # Napari expects (T, Z, Y, X) for a 4D viewer; or (T, Y, X) for
        # a 3D viewer when the timelapse is 2D. We always ship 4 columns
        # — napari is fine with extra dims as long as they're consistent
        # with the image layer.
        out[cls] = rows[["t", "z", "y", "x"]].values.astype(np.float64)
    return out


# ─────────────────────────────────────────────── napari widget
def inception_cellfate_widget():
    """Return a Qt widget that exposes the Inception prediction +
    metrics pipeline as a napari dock plugin."""

    # ----- magicgui input forms ---------------------------------------
    @magicgui(
        label_head=dict(
            widget_type="Label",
            label=f'<h1> <img src="{_abs_resource("resources/kapoorlogo.png")}"> </h1>',
            value="<h5>Inception Cell-Fate Predictor</h5>",
        ),
        model_dir=dict(
            widget_type="FileEdit",
            mode="d",
            label="Model directory (.ckpt + training_config.json)",
        ),
        xml_path=dict(
            widget_type="FileEdit",
            mode="r",
            filter="*.xml",
            label="TrackMate / NapaTrackMater XML",
        ),
        gt_dir=dict(
            widget_type="FileEdit",
            mode="d",
            label="GT CSV directory (optional)",
        ),
        tracklet_length=dict(
            widget_type="SpinBox",
            label="Tracklet length (min frames)",
            min=25,
            max=400,
            value=25,
        ),
        time_window_start=dict(
            widget_type="SpinBox",
            label="Time window start",
            min=0,
            max=10000,
            value=0,
        ),
        time_window_end=dict(
            widget_type="SpinBox",
            label="Time window end (-1 = last)",
            min=-1,
            max=10000,
            value=-1,
        ),
        run_button=dict(widget_type="PushButton", text="Run prediction"),
        layout="vertical",
        persist=True,
        call_button=False,
    )
    def plugin_inputs(
        viewer: napari.Viewer,
        label_head,
        model_dir,
        xml_path,
        gt_dir,
        tracklet_length,
        time_window_start,
        time_window_end,
        run_button,
    ):
        pass

    # ----- metrics tab (confusion matrix + per-class GT yield table) --
    fig = plt.Figure(figsize=(5, 5), tight_layout=True)
    canvas = FigureCanvas(fig)
    canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    gt_yield_table = QTableWidget(0, 4)
    gt_yield_table.setHorizontalHeaderLabels(
        ["class", "GT rows", "Resolved", "Yield %"]
    )

    pred_count_table = QTableWidget(0, 2)
    pred_count_table.setHorizontalHeaderLabels(["class", "Tracks"])

    accuracy_label = QLabel("Accuracy: —")
    accuracy_label.setStyleSheet("font-weight: 600; font-size: 13px;")

    metrics_widget = QWidget()
    metrics_layout = QVBoxLayout(metrics_widget)
    metrics_layout.addWidget(accuracy_label)
    metrics_layout.addWidget(QLabel("Per-class GT → XML resolution"))
    metrics_layout.addWidget(gt_yield_table)
    metrics_layout.addWidget(QLabel("Predicted tracks per class"))
    metrics_layout.addWidget(pred_count_table)
    metrics_layout.addWidget(QLabel("Confusion matrix"))
    metrics_layout.addWidget(canvas)

    # ----- result holders for the runner closures --------------------
    state: dict = {
        "predictions": {},
        "features_df": pd.DataFrame(),
        "gt_df": pd.DataFrame(),
        "class_map": {},
    }

    def _refresh_metrics(
        gt_df: pd.DataFrame,
        gt_stats: dict,
        predictions: Dict[int, str],
        class_color: Dict[str, str],
    ):
        # Pred-track-count table
        pred_count_table.setRowCount(0)
        if predictions:
            counts: Dict[str, int] = {}
            for cls in predictions.values():
                counts[cls] = counts.get(cls, 0) + 1
            for i, (cls, n) in enumerate(sorted(counts.items())):
                pred_count_table.insertRow(i)
                item_cls = QTableWidgetItem(cls)
                item_cls.setForeground(_qcolor(class_color.get(cls)))
                pred_count_table.setItem(i, 0, item_cls)
                pred_count_table.setItem(i, 1, QTableWidgetItem(str(n)))

        # GT yield table
        gt_yield_table.setRowCount(0)
        for i, (cls, v) in enumerate(
            (k, v) for k, v in gt_stats.items() if k != "overall"
        ):
            gt_yield_table.insertRow(i)
            item_cls = QTableWidgetItem(cls)
            item_cls.setForeground(_qcolor(class_color.get(cls)))
            gt_yield_table.setItem(i, 0, item_cls)
            gt_yield_table.setItem(i, 1, QTableWidgetItem(str(v["total"])))
            gt_yield_table.setItem(i, 2, QTableWidgetItem(str(v["matched"])))
            pct = (100.0 * v["matched"] / v["total"]) if v["total"] else 0.0
            gt_yield_table.setItem(i, 3, QTableWidgetItem(f"{pct:.1f}"))

        # Accuracy + confusion matrix
        fig.clear()
        if gt_df.empty or not predictions:
            accuracy_label.setText("Accuracy: — (no GT provided)")
            canvas.draw_idle()
            return

        pred_df = pd.DataFrame(
            [{"Track_ID": int(tid), "class": c} for tid, c in predictions.items()]
        )
        merged = gt_df.merge(
            pred_df, on="Track_ID", suffixes=("_gt", "_pred")
        )
        if merged.empty:
            accuracy_label.setText(
                "Accuracy: — (no GT ↔ prediction overlap on Track_ID)"
            )
            canvas.draw_idle()
            return

        from sklearn.metrics import confusion_matrix

        labels = sorted(set(merged["class_gt"]) | set(merged["class_pred"]))
        cm = confusion_matrix(
            merged["class_gt"], merged["class_pred"], labels=labels
        )
        correct = int((merged["class_gt"] == merged["class_pred"]).sum())
        acc = correct / len(merged)
        overall = gt_stats.get("overall", {"matched": 0, "total": 0})
        accuracy_label.setText(
            f"Accuracy: {acc:.3f}  ·  {len(merged)} matched tracks  "
            f"({overall['matched']} / {overall['total']} GT entries "
            "resolved to XML)"
        )

        ax = fig.add_subplot(111)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground truth")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=9,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        canvas.draw_idle()

    def _qcolor(hexstr: Optional[str]):
        """str → QColor; defaults to black when missing."""
        from qtpy.QtGui import QColor

        return QColor(hexstr or "#000000")

    # ----- run handler ------------------------------------------------
    @functools.wraps(_predict_cellfate)
    def _run():
        viewer: napari.Viewer = plugin_inputs.viewer.value
        model_path = Path(plugin_inputs.model_dir.value)
        xml_path = Path(plugin_inputs.xml_path.value)
        gt_dir = (
            Path(plugin_inputs.gt_dir.value)
            if plugin_inputs.gt_dir.value
            else None
        )

        if not model_path.is_dir():
            napari.utils.notifications.show_error(
                f"Model directory not found: {model_path}"
            )
            return
        if not xml_path.is_file():
            napari.utils.notifications.show_error(
                f"XML file not found: {xml_path}"
            )
            return

        with napari.utils.progress(total=4) as pbar:
            pbar.set_description("Loading model")
            module, arch = _load_inception_model(model_path)
            pbar.update(1)

            pbar.set_description("Running prediction")
            tw = (
                int(plugin_inputs.time_window_start.value),
                int(plugin_inputs.time_window_end.value),
            )
            features_df, predictions, class_map = _predict_cellfate(
                xml_path=xml_path,
                module=module,
                arch=arch,
                tracklet_length=int(plugin_inputs.tracklet_length.value),
                time_window=tw,
            )
            state["predictions"] = predictions
            state["features_df"] = features_df
            state["class_map"] = class_map
            pbar.update(1)

            pbar.set_description("Resolving GT (if provided)")
            gt_df, gt_stats = pd.DataFrame(), {"overall": {"total": 0, "matched": 0}}
            if gt_dir is not None and gt_dir.is_dir():
                gt_csvs = sorted(gt_dir.glob("*.csv"))
                if gt_csvs:
                    gt_df, gt_stats = _resolve_gt_track_ids(xml_path, gt_csvs)
            state["gt_df"] = gt_df
            pbar.update(1)

            pbar.set_description("Updating viewer + metrics")
            color_map = _class_color_map(list(set(predictions.values())))
            # Drop any existing layers we previously added before adding
            # the new batch — otherwise reruns pile up named layers.
            for layer_name in list(viewer.layers):
                if str(layer_name.name).startswith("Pred · "):
                    viewer.layers.remove(layer_name)

            points_by_class = _build_prediction_points(
                features_df, predictions, xml_path
            )
            for cls, pts in points_by_class.items():
                viewer.add_points(
                    pts,
                    name=f"Pred · {cls}",
                    size=12,
                    border_color=color_map.get(cls, "yellow"),
                    face_color="transparent",
                    symbol="o",
                    out_of_slice_display=True,
                )

            # GT layer too — one per class, '+' markers.
            for layer_name in list(viewer.layers):
                if str(layer_name.name).startswith("GT · "):
                    viewer.layers.remove(layer_name)
            if not gt_df.empty:
                for cls, sub in gt_df.groupby("class"):
                    pts = sub[["frame", "z", "y", "x"]].values.astype(
                        np.float64
                    )
                    viewer.add_points(
                        pts,
                        name=f"GT · {cls}",
                        size=14,
                        border_color=color_map.get(cls, "lime"),
                        face_color="transparent",
                        symbol="cross",
                        out_of_slice_display=True,
                    )

            _refresh_metrics(gt_df, gt_stats, predictions, color_map)
            pbar.update(1)

        napari.utils.notifications.show_info(
            f"Predicted {len(predictions)} tracks across "
            f"{len(set(predictions.values()))} classes."
        )

    # Wire the magicgui PushButton at the bottom to our runner.
    plugin_inputs.run_button.clicked.connect(_run)

    # ----- top-level widget ------------------------------------------
    tabs = QTabWidget()
    tabs.addTab(plugin_inputs.native, "Inputs")
    tabs.addTab(metrics_widget, "Metrics")

    container = QWidget()
    layout = QVBoxLayout(container)
    layout.addWidget(tabs)
    return container

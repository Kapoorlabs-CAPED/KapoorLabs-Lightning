"""
ONEAT Event Visualizer and Annotator
Napari plugin for visualizing and adding event annotations
Made by KapoorLabs, 2024
"""

import functools
import os
from pathlib import Path
from typing import List
from glob import glob

import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
from magicgui import magicgui
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from psygnal import Signal
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QSizePolicy,
    QSlider,
    QLabel,
    QPushButton,
    QDoubleSpinBox,
)
from tifffile import imread


SLIDER_STEPS = 5000
DEFAULT_START_PROB = 0.0
DEFAULT_THRESHOLD = 0.7


def plugin_wrapper_oneat_visualizer():

    DEBUG = False
    current_raw_image = None
    current_seg_image = None
    current_csv_data = None
    current_event_name = None
    current_csv_path = None
    clicked_points = []
    raw_files = []
    seg_files = []
    is_loading = False
    has_confidence = False
    has_boxes = False
    event_threshold = DEFAULT_THRESHOLD

    def abspath(root, relpath):
        root = Path(root)
        if root.is_dir():
            path = root / relpath
        else:
            path = root.parent / relpath
        return str(path.absolute())

    kapoorlogo = abspath(__file__, "resources/kapoorlogo.png")

    def change_handler(*widgets, init=False, debug=DEBUG):
        def decorator_change_handler(handler):
            @functools.wraps(handler)
            def wrapper(*args):
                source = Signal.sender()
                emitter = Signal.current_emitter()
                if debug:
                    print(f"{str(emitter.name).upper()}: {source.name}")
                return handler(*args)

            for widget in widgets:
                widget.changed.connect(wrapper)
                if init:
                    widget.changed(widget.value)
            return wrapper

        return decorator_change_handler

    @magicgui(
        label_head=dict(
            widget_type="Label",
            label=f'<h1> <img src="{kapoorlogo}"> </h1>',
            value="<h5>ONEAT Event Visualizer & Annotator</h5>",
        ),
        event_selector=dict(
            widget_type="RadioButtons",
            label="Select Event Type",
            choices=[("mitosis", "mitosis"), ("normal", "normal")],
            value="mitosis",
        ),
        add_point_mode=dict(
            widget_type="CheckBox",
            label="Add Point Mode (Double-click)",
            value=False,
        ),
        status_label=dict(
            widget_type="Label",
            label="Status:",
            value="",
        ),
        layout="vertical",
        persist=True,
        call_button=False,
    )
    def plugin(
        viewer: napari.Viewer,
        label_head,
        event_selector,
        add_point_mode,
        status_label,
    ) -> List[napari.types.LayerDataTuple]:
        pass

    @magicgui(
        raw_dir=dict(
            widget_type="FileEdit",
            mode="d",
            label="Raw Images Directory",
        ),
        seg_dir=dict(
            widget_type="FileEdit",
            mode="d",
            label="Seg Images Directory (Optional)",
        ),
        csv_dir=dict(
            widget_type="FileEdit",
            mode="d",
            label="CSV Files Directory",
        ),
        load_data_button=dict(
            widget_type="PushButton",
            text="Load Data",
        ),
        layout="vertical",
        persist=False,
        call_button=False,
    )
    def plugin_data(
        raw_dir,
        seg_dir,
        csv_dir,
        load_data_button,
    ):
        pass

    @magicgui(
        raw_image_selector=dict(
            widget_type="ComboBox",
            label="Select Raw Image",
            choices=[],
        ),
        seg_image_selector=dict(
            widget_type="ComboBox",
            label="Select Seg Image",
            choices=["None"],
        ),
        save_csv_button=dict(
            widget_type="PushButton",
            text="Save Updated CSV",
        ),
        layout="vertical",
        persist=False,
        call_button=False,
    )
    def plugin_select(
        raw_image_selector,
        seg_image_selector,
        save_csv_button,
    ):
        pass

    # ── Qt-based confidence threshold controls ──
    # Container widget for slider + spinbox + label + save button
    threshold_container = QWidget()
    threshold_layout = QVBoxLayout()
    threshold_container.setLayout(threshold_layout)

    # Start prob spinbox row
    startprob_row = QHBoxLayout()
    startprob_label = QLabel("Start Prob:")
    startprob_spinbox = QDoubleSpinBox()
    startprob_spinbox.setValue(DEFAULT_START_PROB)  # 0.0 — full range
    startprob_spinbox.setDecimals(3)
    startprob_spinbox.setRange(0.0, 1.0)
    startprob_spinbox.setSingleStep(0.01)
    startprob_row.addWidget(startprob_label)
    startprob_row.addWidget(startprob_spinbox)
    threshold_layout.addLayout(startprob_row)

    # Slider row
    slider_row = QHBoxLayout()
    slider_label_text = QLabel("Confidence Threshold:")
    score_slider = QSlider(Qt.Horizontal)
    score_slider.setToolTip("Scroll through probability score")
    score_slider.setRange(0, SLIDER_STEPS)
    score_slider.setSingleStep(1)
    score_slider.setTickInterval(1)
    # Set slider to position corresponding to DEFAULT_THRESHOLD (0.7)
    default_slider_pos = int(
        (DEFAULT_THRESHOLD - DEFAULT_START_PROB)
        / (1.0 - DEFAULT_START_PROB)
        * SLIDER_STEPS
    )
    score_slider.setValue(default_slider_pos)
    score_value_label = QLabel(f"{DEFAULT_THRESHOLD:.3f}")
    slider_row.addWidget(slider_label_text)
    slider_row.addWidget(score_slider)
    slider_row.addWidget(score_value_label)
    threshold_layout.addLayout(slider_row)

    # Save high prob button
    save_high_prob_btn = QPushButton("Save High Probability Detections")
    threshold_layout.addWidget(save_high_prob_btn)

    # Hide until confidence data is loaded
    threshold_container.setVisible(False)

    def _slider_to_threshold(slider_value):
        """Convert integer slider value to real probability threshold."""
        start_prob = startprob_spinbox.value()
        return start_prob + (1.0 - start_prob) / SLIDER_STEPS * float(slider_value)

    apply_threshold_btn = QPushButton("Apply Threshold")
    threshold_layout.addWidget(apply_threshold_btn)

    def _on_slider_moved(value):
        """Only update the label text, nothing else."""
        real_value = _slider_to_threshold(value)
        score_value_label.setText(f"{real_value:.3f}")

    score_slider.valueChanged.connect(_on_slider_moved)

    def _apply_threshold():
        """Filter the CSV by threshold, put result in a new green points layer."""
        nonlocal event_threshold
        real_value = _slider_to_threshold(score_slider.value())
        event_threshold = real_value
        score_value_label.setText(f"{real_value:.3f}")
        if is_loading or current_csv_data is None or not has_confidence:
            return

        # Filter the original dataframe
        mask = current_csv_data["score"] >= real_value
        high_prob_df = current_csv_data[mask]
        points = (
            high_prob_df[["t", "z", "y", "x"]].values
            if len(high_prob_df) > 0
            else np.empty((0, 4))
        )

        viewer = plugin.viewer.value

        # Update the existing High Prob Events layer data
        for layer in viewer.layers:
            if layer.name == "High Prob Events":
                layer.data = points
                break

        # Update boxes layer for high-prob events only
        if has_boxes:
            box_df = (
                high_prob_df.dropna(subset=["h", "w", "d"])
                if len(high_prob_df) > 0
                else pd.DataFrame()
            )
            for layer in viewer.layers:
                if layer.name == "High Prob Boxes":
                    if len(box_df) > 0:
                        rectangles = _make_box_rectangles(box_df)
                        layer.data = rectangles
                    else:
                        layer.data = []
                    break

        _overlay_high_prob_on_plot(high_prob_df, real_value)
        plugin.status_label.value = (
            f"{len(high_prob_df)} high-prob detections (threshold >= {real_value:.3f})"
        )

    apply_threshold_btn.clicked.connect(_apply_threshold)

    def _on_save_high_prob():
        """Save detections above current threshold as a new file."""
        if current_csv_data is None or not has_confidence or current_csv_path is None:
            plugin.status_label.value = "No prediction data with confidence to filter"
            return

        filtered_df = current_csv_data[current_csv_data["score"] >= event_threshold]
        if len(filtered_df) == 0:
            plugin.status_label.value = "No detections above current threshold"
            return

        csv_dir = os.path.dirname(current_csv_path)
        csv_stem = os.path.splitext(os.path.basename(current_csv_path))[0]
        threshold_str = f"{event_threshold:.2f}".replace(".", "p")
        output_name = f"{csv_stem}_high_prob_{threshold_str}.csv"
        output_path = os.path.join(csv_dir, output_name)

        filtered_df.to_csv(output_path, index=False)
        plugin.status_label.value = (
            f"Saved {len(filtered_df)} high-prob detections to {output_name}"
        )
        print(f"Saved high probability detections: {output_path}")

    save_high_prob_btn.clicked.connect(_on_save_high_prob)

    # ── Mouse callback for adding points ──
    mouse_callback_registered = False

    def get_event(viewer, event):
        nonlocal clicked_points, current_csv_data, current_event_name

        if not plugin.add_point_mode.value:
            return

        clicked_location = event.position

        # Extract coordinates based on dimensionality
        if len(clicked_location) == 4:  # TZYX
            t, z, y, x = (int(coord) for coord in clicked_location)
        elif len(clicked_location) == 3:  # TYX
            t, y, x = (int(coord) for coord in clicked_location)
            z = 0
        else:
            return

        new_point = {"t": t, "z": z, "y": y, "x": x}
        clicked_points.append(new_point)

        print(f"Added point: t={t}, z={z}, y={y}, x={x}")
        plugin.status_label.value = (
            f"Added point at t={t}, z={z}, y={y}, x={x} ({len(clicked_points)} new)"
        )

        _update_viewer_with_new_points()
        _update_table()

    def _make_box_rectangles(df):
        """Create rectangle corner arrays for napari Shapes layer from detections with h, w, d columns."""
        rectangles = []
        for _, row in df.iterrows():
            t = row["t"]
            z = row["z"]
            y = row["y"]
            x = row["x"]
            h = row["h"]
            w = row["w"]
            half_h = h / 2
            half_w = w / 2
            rect = np.array(
                [
                    [t, z, y - half_h, x - half_w],
                    [t, z, y - half_h, x + half_w],
                    [t, z, y + half_h, x + half_w],
                    [t, z, y + half_h, x - half_w],
                ]
            )
            rectangles.append(rect)
        return rectangles

    def _get_filtered_data():
        """Return current CSV data filtered by confidence threshold if applicable."""
        if current_csv_data is None or len(current_csv_data) == 0:
            return pd.DataFrame(columns=["t", "z", "y", "x"])

        df = current_csv_data.copy()
        if has_confidence and "score" in df.columns:
            df = df[df["score"] >= event_threshold].reset_index(drop=True)
        return df

    def _update_viewer_with_new_points():
        """Update points layer with new clicked points"""
        combined_df = _get_filtered_data()

        if len(clicked_points) > 0:
            combined_df = pd.concat(
                [combined_df, pd.DataFrame(clicked_points)], ignore_index=True
            )

        # Remove old points and shapes layers
        for layer in list(plugin.viewer.value.layers):
            if isinstance(layer, (napari.layers.Points, napari.layers.Shapes)):
                plugin.viewer.value.layers.remove(layer)

        if len(combined_df) > 0:
            points_array = combined_df[["t", "z", "y", "x"]].values

            kwargs = dict(
                name=f"{current_event_name} Events (Updated)",
                face_color="transparent",
                border_color="red",
                size=10,
                ndim=4,
            )

            if has_confidence and "score" in combined_df.columns:
                kwargs["properties"] = {
                    "score": combined_df["score"]
                    .apply(lambda c: f"{c:.3f}" if pd.notna(c) else "")
                    .values
                }
                kwargs["text"] = {
                    "string": "{score}",
                    "size": 8,
                    "color": "yellow",
                    "anchor": "upper_left",
                }

            plugin.viewer.value.add_points(points_array, **kwargs)

            if has_boxes and all(c in combined_df.columns for c in ("h", "w", "d")):
                box_df = combined_df.dropna(subset=["h", "w", "d"])
                if len(box_df) > 0:
                    rectangles = _make_box_rectangles(box_df)
                    plugin.viewer.value.add_shapes(
                        rectangles,
                        shape_type="rectangle",
                        name=f"{current_event_name} Boxes (Updated)",
                        edge_color="cyan",
                        face_color="transparent",
                        edge_width=2,
                        ndim=4,
                    )

    def _update_table():
        """Update table widget with current CSV data"""
        combined_df = _get_filtered_data()

        if len(clicked_points) > 0:
            combined_df = pd.concat(
                [combined_df, pd.DataFrame(clicked_points)], ignore_index=True
            )

        display_cols = [
            c
            for c in ("t", "z", "y", "x", "score", "size", "h", "w", "d")
            if c in combined_df.columns
        ]
        if not display_cols:
            display_cols = ["t", "z", "y", "x"]

        # Block signals to avoid per-cell redraws
        table_widget.blockSignals(True)
        table_widget.setUpdatesEnabled(False)

        table_widget.setColumnCount(len(display_cols))
        table_widget.setHorizontalHeaderLabels([c.upper() for c in display_cols])
        table_widget.setRowCount(len(combined_df))

        if len(combined_df) > 0:
            for row_idx, (_, row) in enumerate(combined_df.iterrows()):
                for col_idx, col in enumerate(display_cols):
                    if col in row.index:
                        val = row[col]
                        if col in ("score", "size", "h", "w", "d"):
                            cell_text = f"{val:.3f}" if pd.notna(val) else ""
                        else:
                            cell_text = str(int(val))
                        table_widget.setItem(
                            row_idx, col_idx, QTableWidgetItem(cell_text)
                        )

        table_widget.setUpdatesEnabled(True)
        table_widget.blockSignals(False)

    def _overlay_high_prob_on_plot(filtered_df, threshold):
        """Add green overlay on existing plot for high-prob detections."""
        if len(plot_fig.axes) == 0:
            return
        ax = plot_fig.axes[0]

        # Remove previous green overlay bars (identified by label)
        bars_to_remove = [
            c
            for c in ax.containers
            if hasattr(c, "get_label") and c.get_label().startswith(">=")
        ]
        for bar in bars_to_remove:
            bar.remove()
        # Also remove from legend
        legend_handles = []
        legend_labels = []
        for h, l in zip(*ax.get_legend_handles_labels()):
            if not l.startswith(">="):
                legend_handles.append(h)
                legend_labels.append(l)

        if len(filtered_df) > 0 and "t" in filtered_df.columns:
            all_df = (
                current_csv_data
                if current_csv_data is not None
                else pd.DataFrame(columns=["t"])
            )
            if len(all_df) > 0:
                t_min = int(all_df["t"].min())
                t_max = int(all_df["t"].max())
                bins = np.arange(t_min, t_max + 2) - 0.5
                bars = ax.hist(
                    filtered_df["t"].values,
                    bins=bins,
                    color="#2ecc71",
                    edgecolor="white",
                    alpha=0.85,
                    label=f">= {threshold:.3f} ({len(filtered_df)})",
                )
                legend_handles.append(bars[2][0])
                legend_labels.append(f">= {threshold:.3f} ({len(filtered_df)})")

        ax.legend(legend_handles, legend_labels, fontsize=8)
        plot_canvas.draw_idle()

    def _update_plot():
        """Update the detections-over-time plot.
        All detections in red, high-prob (above threshold) overlaid in green."""
        plot_fig.clear()
        ax = plot_fig.add_subplot(111)

        all_df = (
            current_csv_data
            if current_csv_data is not None
            else pd.DataFrame(columns=["t"])
        )
        filtered_df = _get_filtered_data()

        if len(all_df) > 0 and "t" in all_df.columns:
            t_min = int(all_df["t"].min())
            t_max = int(all_df["t"].max())
            bins = np.arange(t_min, t_max + 2) - 0.5

            # All detections in red
            ax.hist(
                all_df["t"].values,
                bins=bins,
                color="#e74c3c",
                edgecolor="white",
                alpha=0.5,
                label=f"all ({len(all_df)})",
            )

            # High-prob detections in green on top
            if has_confidence and len(filtered_df) > 0:
                ax.hist(
                    filtered_df["t"].values,
                    bins=bins,
                    color="#2ecc71",
                    edgecolor="white",
                    alpha=0.85,
                    label=f">= {event_threshold:.3f} ({len(filtered_df)})",
                )

            ax.set_xlabel("Timepoint")
            ax.set_ylabel("Detections")
            title = f"{current_event_name} detections"
            if has_confidence:
                title += f" | threshold >= {event_threshold:.3f}"
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8)
        else:
            ax.set_title("No detections", fontsize=10)
            ax.set_xlabel("Timepoint")
            ax.set_ylabel("Detections")

        plot_fig.tight_layout()
        plot_canvas.draw()

    @change_handler(plugin_data.raw_dir)
    def _on_raw_dir_changed(value):
        """When raw directory is selected, default csv_dir to the same path."""
        if value and str(value) and not str(plugin_data.csv_dir.value):
            plugin_data.csv_dir.value = value

    @change_handler(plugin_data.load_data_button)
    def _on_load_data_clicked(value):
        nonlocal raw_files, seg_files, mouse_callback_registered, is_loading

        if not mouse_callback_registered and plugin.viewer.value is not None:
            plugin.viewer.value.mouse_double_click_callbacks.append(get_event)
            mouse_callback_registered = True

        raw_directory = (
            str(plugin_data.raw_dir.value) if plugin_data.raw_dir.value else None
        )
        seg_directory = (
            str(plugin_data.seg_dir.value) if plugin_data.seg_dir.value else None
        )
        csv_directory = (
            str(plugin_data.csv_dir.value) if plugin_data.csv_dir.value else None
        )

        if not raw_directory or not csv_directory:
            plugin.status_label.value = "Error: Must select raw and CSV directories"
            return

        raw_files = sorted(glob(os.path.join(raw_directory, "*.tif")))
        if seg_directory:
            seg_files = sorted(glob(os.path.join(seg_directory, "*.tif")))
        else:
            seg_files = []

        if len(raw_files) == 0:
            plugin.status_label.value = "Error: No .tif files found in raw directory"
            return

        is_loading = True
        try:
            plugin_select.raw_image_selector.choices = [
                os.path.basename(f) for f in raw_files
            ]
            if seg_files:
                plugin_select.seg_image_selector.choices = ["None"] + [
                    os.path.basename(f) for f in seg_files
                ]
            else:
                plugin_select.seg_image_selector.choices = ["None"]

            plugin.status_label.value = f"Loaded: {len(raw_files)} raw images"
        finally:
            is_loading = False

        _load_current_selection()

    def _load_current_selection():
        """Load currently selected raw image, seg image, and CSV"""
        nonlocal current_raw_image, current_seg_image, current_csv_data, current_event_name, current_csv_path, clicked_points, is_loading, has_confidence, has_boxes

        if is_loading or len(raw_files) == 0:
            return

        is_loading = True
        try:
            raw_selected = plugin_select.raw_image_selector.value
            seg_selected = plugin_select.seg_image_selector.value
            selected_event = plugin.event_selector.value

            if not raw_selected:
                return

            raw_path = None
            for f in raw_files:
                if os.path.basename(f) == raw_selected:
                    raw_path = f
                    break

            if not raw_path:
                return

            current_raw_image = imread(raw_path)

            current_seg_image = None
            if seg_selected and seg_selected != "None":
                for f in seg_files:
                    if os.path.basename(f) == seg_selected:
                        current_seg_image = imread(f)
                        break

            raw_basename = os.path.basename(raw_path)
            image_name = os.path.splitext(raw_basename)[0]
            csv_directory = str(plugin_data.csv_dir.value)
            csv_pattern = f"oneat_{selected_event}_{image_name}.csv"
            csv_path = os.path.join(csv_directory, csv_pattern)

            print(f"Looking for CSV: {csv_path}")

            if os.path.exists(csv_path):
                current_csv_data = pd.read_csv(csv_path)
                current_csv_data.columns = [
                    col.lower() for col in current_csv_data.columns
                ]
                if "time" in current_csv_data.columns:
                    current_csv_data = current_csv_data.rename(columns={"time": "t"})
                # Normalize old 'confidence' column name to 'score'
                if "confidence" in current_csv_data.columns:
                    current_csv_data = current_csv_data.rename(
                        columns={"confidence": "score"}
                    )
                has_confidence = "score" in current_csv_data.columns
                has_boxes = all(c in current_csv_data.columns for c in ("h", "w", "d"))
                current_csv_path = csv_path
                print(
                    f"Loaded CSV with {len(current_csv_data)} events (confidence: {has_confidence}, boxes: {has_boxes})"
                )
            else:
                current_csv_data = pd.DataFrame(columns=["t", "z", "y", "x"])
                has_confidence = False
                has_boxes = False
                current_csv_path = csv_path
                print("No CSV found, created empty DataFrame")

            current_event_name = selected_event
            clicked_points = []

            # Show/hide confidence controls
            threshold_container.setVisible(has_confidence)

            _update_viewer()
            _update_table()
            _update_plot()

            csv_status = "existing" if os.path.exists(csv_path) else "new"
            extra = ""
            if has_confidence:
                extra = f", score range: [{current_csv_data['score'].min():.3f}, {current_csv_data['score'].max():.3f}]"
            plugin.status_label.value = f"{image_name} - {selected_event} ({len(current_csv_data)} events, {csv_status}{extra})"
        finally:
            is_loading = False

    def _update_viewer():
        """Update napari viewer with current images and points"""
        plugin.viewer.value.layers.clear()

        plugin.viewer.value.add_image(
            current_raw_image, name="Raw Image", colormap="gray"
        )

        if current_seg_image is not None:
            plugin.viewer.value.add_labels(current_seg_image, name="Segmentation")

        filtered_df = _get_filtered_data()

        if len(filtered_df) > 0:
            points_array = filtered_df[["t", "z", "y", "x"]].values
            print(f"Adding {len(points_array)} points")
            print(f"Image shape: {current_raw_image.shape}")

            kwargs = dict(
                name=f"{current_event_name} Events",
                face_color="transparent",
                border_color="red",
                size=10,
                ndim=4,
            )

            if has_confidence and "score" in filtered_df.columns:
                kwargs["properties"] = {
                    "score": filtered_df["score"].apply(lambda c: f"{c:.3f}").values
                }
                kwargs["text"] = {
                    "string": "{score}",
                    "size": 8,
                    "color": "yellow",
                    "anchor": "upper_left",
                }

            plugin.viewer.value.add_points(points_array, **kwargs)

            # Create High Prob Events layer with same data (green filled)
            # Threshold changes will just swap its data
            if has_confidence:
                high_prob_mask = current_csv_data["score"] >= event_threshold
                high_prob_df = current_csv_data[high_prob_mask]
                high_prob_points = (
                    high_prob_df[["t", "z", "y", "x"]].values
                    if len(high_prob_df) > 0
                    else np.empty((0, 4))
                )

                plugin.viewer.value.add_points(
                    high_prob_points,
                    name="High Prob Events",
                    border_color="green",
                    face_color="transparent",
                    size=12,
                    ndim=4,
                )

                # Boxes only for high-prob events, updated with threshold
                if has_boxes and len(high_prob_df) > 0:
                    box_df = high_prob_df.dropna(subset=["h", "w", "d"])
                    if len(box_df) > 0:
                        rectangles = _make_box_rectangles(box_df)
                        plugin.viewer.value.add_shapes(
                            rectangles,
                            shape_type="rectangle",
                            name="High Prob Boxes",
                            edge_color="cyan",
                            face_color="transparent",
                            edge_width=2,
                            ndim=4,
                        )
                    else:
                        # Empty shapes layer so we can swap data later
                        plugin.viewer.value.add_shapes(
                            [],
                            shape_type="rectangle",
                            name="High Prob Boxes",
                            edge_color="cyan",
                            face_color="transparent",
                            edge_width=2,
                            ndim=4,
                        )
                elif has_boxes:
                    plugin.viewer.value.add_shapes(
                        [],
                        shape_type="rectangle",
                        name="High Prob Boxes",
                        edge_color="cyan",
                        face_color="transparent",
                        edge_width=2,
                        ndim=4,
                    )

    @change_handler(
        plugin_select.raw_image_selector,
        plugin_select.seg_image_selector,
        plugin.event_selector,
    )
    def _on_selection_changed(value):
        """Auto-load when selections change"""
        if not is_loading and len(raw_files) > 0:
            _load_current_selection()

    @change_handler(plugin_select.save_csv_button)
    def _on_save_csv_clicked(value):
        """Save updated CSV with new points"""
        nonlocal clicked_points, current_csv_data

        if len(clicked_points) == 0:
            plugin.status_label.value = "No new points to save"
            return

        raw_selected = plugin_select.raw_image_selector.value
        if not raw_selected:
            plugin.status_label.value = "Error: No image selected"
            return

        image_name = os.path.splitext(raw_selected)[0]

        csv_directory = str(plugin_data.csv_dir.value)
        output_csv = os.path.join(
            csv_directory, f"oneat_{current_event_name}_{image_name}.csv"
        )

        combined_df = (
            current_csv_data.copy()
            if current_csv_data is not None and len(current_csv_data) > 0
            else pd.DataFrame(columns=["t", "z", "y", "x"])
        )

        if len(clicked_points) > 0:
            combined_df = pd.concat(
                [combined_df, pd.DataFrame(clicked_points)], ignore_index=True
            )

        combined_df.columns = [col.lower() for col in combined_df.columns]
        combined_df = combined_df.sort_values("t").reset_index(drop=True)

        combined_df.to_csv(output_csv, index=False)

        plugin.status_label.value = (
            f"Saved {len(combined_df)} points to {os.path.basename(output_csv)}"
        )
        print(f"Saved CSV: {output_csv}")

        current_csv_data = combined_df.copy()
        clicked_points = []

        _update_table()

    # Setup UI layout
    plugin.label_head.native.setSizePolicy(
        QSizePolicy.MinimumExpanding, QSizePolicy.Fixed
    )

    # Create tabs
    tabs = QTabWidget()

    # Data input tab
    data_tab = QWidget()
    data_tab_layout = QVBoxLayout()
    data_tab.setLayout(data_tab_layout)
    data_tab_layout.addWidget(plugin_data.native)
    tabs.addTab(data_tab, "Input Data")

    # Selection tab
    select_tab = QWidget()
    select_tab_layout = QVBoxLayout()
    select_tab.setLayout(select_tab_layout)
    select_tab_layout.addWidget(plugin_select.native)
    tabs.addTab(select_tab, "Select Images")

    # Table tab
    table_widget = QTableWidget()
    table_widget.setColumnCount(4)
    table_widget.setHorizontalHeaderLabels(["T", "Z", "Y", "X"])
    tabs.addTab(table_widget, "Event Table")

    # Plot tab - detections over time
    plot_fig = plt.Figure(figsize=(5, 3), dpi=100)
    plot_canvas = FigureCanvas(plot_fig)
    plot_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    tabs.addTab(plot_canvas, "Detection Plot")

    # Add tabs to main plugin layout
    plugin.native.layout().addWidget(tabs)

    # Add threshold controls below tabs (hidden until confidence data loaded)
    plugin.native.layout().addWidget(threshold_container)

    return plugin


# Create the widget
oneat_visualizer_widget = plugin_wrapper_oneat_visualizer()

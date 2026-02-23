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

import napari
import numpy as np
import pandas as pd
from magicgui import magicgui
from magicgui import widgets as mw
from psygnal import Signal
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QSizePolicy,
)
from tifffile import imread


def plugin_wrapper_oneat_visualizer():

    DEBUG = False
    current_raw_image = None
    current_seg_image = None
    current_csv_data = None
    current_event_name = None
    clicked_points = []
    raw_files = []
    seg_files = []

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
            value='<h5>ONEAT Event Visualizer & Annotator</h5>',
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

    # Mouse callback for adding points
    mouse_callback_registered = False

    def get_event(viewer, event):
        nonlocal clicked_points, current_csv_data, current_event_name

        if not plugin.add_point_mode.value:
            return

        clicked_location = event.position

        # Extract coordinates based on dimensionality
        if len(clicked_location) == 4:  # TZYX
            t, z, y, x = [int(coord) for coord in clicked_location]
        elif len(clicked_location) == 3:  # TYX
            t, y, x = [int(coord) for coord in clicked_location]
            z = 0
        else:
            return

        # Add to clicked points (always use lowercase)
        new_point = {'t': t, 'z': z, 'y': y, 'x': x}
        clicked_points.append(new_point)

        print(f"Added point: t={t}, z={z}, y={y}, x={x}")
        plugin.status_label.value = f"Added point at t={t}, z={z}, y={y}, x={x} ({len(clicked_points)} new)"

        # Update viewer and table
        _update_viewer_with_new_points()
        _update_table()

    def _update_viewer_with_new_points():
        """Update points layer with new clicked points"""
        nonlocal current_csv_data

        # Combine existing and new points
        combined_df = current_csv_data.copy() if current_csv_data is not None and len(current_csv_data) > 0 else pd.DataFrame(columns=['t', 'z', 'y', 'x'])

        if len(clicked_points) > 0:
            combined_df = pd.concat([
                combined_df,
                pd.DataFrame(clicked_points)
            ], ignore_index=True)

        # Remove old points layer
        for layer in plugin.viewer.value.layers:
            if isinstance(layer, napari.layers.Points):
                plugin.viewer.value.layers.remove(layer)

        # Add updated points
        if len(combined_df) > 0:
            points_array = combined_df[['t', 'z', 'y', 'x']].values
            plugin.viewer.value.add_points(
                points_array,
                name=f"{current_event_name} Events (Updated)",
                face_color="red",
                border_color="white",
                size=10,
                ndim=4,
            )

    def _update_table():
        """Update table widget with current CSV data"""
        nonlocal current_csv_data

        combined_df = current_csv_data.copy() if current_csv_data is not None and len(current_csv_data) > 0 else pd.DataFrame(columns=['t', 'z', 'y', 'x'])

        # Add new clicked points
        if len(clicked_points) > 0:
            combined_df = pd.concat([
                combined_df,
                pd.DataFrame(clicked_points)
            ], ignore_index=True)

        table_widget.setRowCount(len(combined_df))

        if len(combined_df) > 0:
            for row_idx, (_, row) in enumerate(combined_df.iterrows()):
                table_widget.setItem(row_idx, 0, QTableWidgetItem(str(int(row['t']))))
                table_widget.setItem(row_idx, 1, QTableWidgetItem(str(int(row['z']))))
                table_widget.setItem(row_idx, 2, QTableWidgetItem(str(int(row['y']))))
                table_widget.setItem(row_idx, 3, QTableWidgetItem(str(int(row['x']))))

    @change_handler(plugin_data.load_data_button)
    def _on_load_data_clicked(value):
        nonlocal raw_files, seg_files, mouse_callback_registered

        # Register mouse callback if not already done
        if not mouse_callback_registered and plugin.viewer.value is not None:
            plugin.viewer.value.mouse_double_click_callbacks.append(get_event)
            mouse_callback_registered = True

        # Get directories
        raw_directory = str(plugin_data.raw_dir.value) if plugin_data.raw_dir.value else None
        seg_directory = str(plugin_data.seg_dir.value) if plugin_data.seg_dir.value else None
        csv_directory = str(plugin_data.csv_dir.value) if plugin_data.csv_dir.value else None

        if not raw_directory or not csv_directory:
            plugin.status_label.value = "Error: Must select raw and CSV directories"
            return

        # Load file lists
        raw_files = sorted(glob(os.path.join(raw_directory, "*.tif")))
        if seg_directory:
            seg_files = sorted(glob(os.path.join(seg_directory, "*.tif")))
        else:
            seg_files = []

        if len(raw_files) == 0:
            plugin.status_label.value = "Error: No .tif files found in raw directory"
            return

        # Update selectors
        plugin_select.raw_image_selector.choices = [os.path.basename(f) for f in raw_files]
        if seg_files:
            plugin_select.seg_image_selector.choices = ["None"] + [os.path.basename(f) for f in seg_files]
        else:
            plugin_select.seg_image_selector.choices = ["None"]

        plugin.status_label.value = f"Loaded: {len(raw_files)} raw images"

        # Auto-load first image
        _load_current_selection()

    def _load_current_selection():
        """Load currently selected raw image, seg image, and CSV"""
        nonlocal current_raw_image, current_seg_image, current_csv_data, current_event_name, clicked_points

        if len(raw_files) == 0:
            return

        raw_selected = plugin_select.raw_image_selector.value
        seg_selected = plugin_select.seg_image_selector.value
        selected_event = plugin.event_selector.value

        if not raw_selected:
            return

        # Find raw file path
        raw_path = None
        for f in raw_files:
            if os.path.basename(f) == raw_selected:
                raw_path = f
                break

        if not raw_path:
            return

        # Load raw image
        current_raw_image = imread(raw_path)

        # Load seg image if selected
        current_seg_image = None
        if seg_selected and seg_selected != "None":
            for f in seg_files:
                if os.path.basename(f) == seg_selected:
                    current_seg_image = imread(f)
                    break

        # Load CSV
        raw_basename = os.path.basename(raw_path)
        image_name = os.path.splitext(raw_basename)[0]
        csv_directory = str(plugin_data.csv_dir.value)
        csv_pattern = f"oneat_{selected_event}_{image_name}.csv"
        csv_path = os.path.join(csv_directory, csv_pattern)

        print(f"Looking for CSV: {csv_path}")

        if os.path.exists(csv_path):
            current_csv_data = pd.read_csv(csv_path)
            # Normalize column names to lowercase (agnostic to input case)
            current_csv_data.columns = [col.lower() for col in current_csv_data.columns]
            # Rename time column if needed
            if 'time' in current_csv_data.columns:
                current_csv_data = current_csv_data.rename(columns={'time': 't'})
            print(f"Loaded CSV with {len(current_csv_data)} events")
        else:
            current_csv_data = pd.DataFrame(columns=['t', 'z', 'y', 'x'])
            print(f"No CSV found, created empty DataFrame")

        current_event_name = selected_event
        clicked_points = []

        # Update viewer
        _update_viewer()

        # Update table
        _update_table()

        csv_status = "existing" if os.path.exists(csv_path) else "new"
        plugin.status_label.value = f"{image_name} - {selected_event} ({len(current_csv_data)} events, {csv_status})"

    def _update_viewer():
        """Update napari viewer with current images and points"""
        nonlocal current_csv_data

        # Clear layers
        plugin.viewer.value.layers.clear()

        # Add raw image
        plugin.viewer.value.add_image(current_raw_image, name="Raw Image", colormap="gray")

        # Add seg image if available
        if current_seg_image is not None:
            plugin.viewer.value.add_labels(current_seg_image, name="Segmentation")

        # Add points
        if current_csv_data is not None and len(current_csv_data) > 0:
            points_array = current_csv_data[['t', 'z', 'y', 'x']].values
            print(f"Adding {len(points_array)} points")
            print(f"Image shape: {current_raw_image.shape}")

            plugin.viewer.value.add_points(
                points_array,
                name=f"{current_event_name} Events",
                face_color="red",
                border_color="white",
                size=10,
                ndim=4,
            )

    @change_handler(plugin_select.raw_image_selector, plugin_select.seg_image_selector, plugin.event_selector)
    def _on_selection_changed(value):
        """Auto-load when selections change"""
        if len(raw_files) > 0:
            _load_current_selection()

    @change_handler(plugin_select.save_csv_button)
    def _on_save_csv_clicked(value):
        """Save updated CSV with new points"""
        nonlocal clicked_points, current_csv_data

        if len(clicked_points) == 0:
            plugin.status_label.value = "No new points to save"
            return

        # Get current image name
        raw_selected = plugin_select.raw_image_selector.value
        if not raw_selected:
            plugin.status_label.value = "Error: No image selected"
            return

        image_name = os.path.splitext(raw_selected)[0]

        # Create output CSV filename
        csv_directory = str(plugin_data.csv_dir.value)
        output_csv = os.path.join(
            csv_directory,
            f"oneat_{current_event_name}_{image_name}.csv"
        )

        # Combine existing and new points
        combined_df = current_csv_data.copy() if current_csv_data is not None and len(current_csv_data) > 0 else pd.DataFrame(columns=['t', 'z', 'y', 'x'])

        if len(clicked_points) > 0:
            combined_df = pd.concat([
                combined_df,
                pd.DataFrame(clicked_points)
            ], ignore_index=True)

        # Ensure lowercase column names (always save with lowercase)
        combined_df.columns = [col.lower() for col in combined_df.columns]

        # Sort by time
        combined_df = combined_df.sort_values('t').reset_index(drop=True)

        # Save to CSV with lowercase columns
        combined_df.to_csv(output_csv, index=False)

        plugin.status_label.value = f"Saved {len(combined_df)} points to {os.path.basename(output_csv)}"
        print(f"Saved CSV: {output_csv}")

        # Update current data and reset clicked points
        current_csv_data = combined_df.copy()
        clicked_points = []

        # Update table
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
    table_widget.setHorizontalHeaderLabels(['T', 'Z', 'Y', 'X'])
    tabs.addTab(table_widget, "Event Table")

    # Add tabs to main plugin layout
    plugin.native.layout().addWidget(tabs)

    return plugin


# Create the widget
oneat_visualizer_widget = plugin_wrapper_oneat_visualizer()

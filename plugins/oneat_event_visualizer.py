"""
ONEAT Event Visualizer and Annotator
Napari plugin for visualizing and adding event annotations
Made by KapoorLabs, 2024
"""

import functools
import os
from pathlib import Path
from typing import List

import napari
import numpy as np
import pandas as pd
from magicgui import magicgui
from magicgui import widgets as mw
from psygnal import Signal
from qtpy.QtWidgets import QVBoxLayout, QWidget
from tifffile import imread
from glob import glob


def plugin_wrapper_oneat_visualizer():

    DEBUG = False
    current_raw_image = None
    current_seg_image = None
    current_csv_data = None
    current_event_name = None
    current_timepoint = 0
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

    def _raise(e):
        if isinstance(e, BaseException):
            raise e
        else:
            raise ValueError(e)

    def get_data(image, debug=DEBUG):
        if hasattr(image, 'data'):
            image = image.data[0] if image.multiscale else image.data
        if debug:
            print("Image loaded")
        return np.asarray(image)

    def load_csv_file(csv_path, event_name):
        """Load CSV file and filter by event name if present in filename"""
        if not os.path.exists(csv_path):
            return pd.DataFrame(columns=['t', 'z', 'y', 'x'])

        df = pd.read_csv(csv_path)

        # Rename time column if needed
        if 'time' in df.columns:
            df = df.rename(columns={'time': 't'})

        # Ensure required columns exist
        required_cols = ['t', 'z', 'y', 'x']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0

        return df

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
        raw_image_selector=dict(
            widget_type="ComboBox",
            label="Select Raw Image",
            choices=[],
            visible=False,
        ),
        seg_image_selector=dict(
            widget_type="ComboBox",
            label="Select Seg Image (Optional)",
            choices=[],
            visible=False,
        ),
        event_selector=dict(
            widget_type="RadioButtons",
            label="Select Event Type",
            choices=[("mitosis", "mitosis"), ("normal", "normal")],
            value="mitosis",
            visible=False,
        ),
        add_point_mode=dict(
            widget_type="CheckBox",
            label="Add Point Mode",
            value=False,
            visible=False,
        ),
        save_csv_button=dict(
            widget_type="PushButton",
            text="Save Updated CSV",
            visible=False,
        ),
        status_label=dict(
            widget_type="Label",
            label="Status:",
            value="",
        ),
        progress_bar=dict(label=" ", min=0, max=0, visible=False),
        layout="vertical",
        persist=True,
        call_button=False,
    )
    def plugin(
        viewer: napari.Viewer,
        label_head,
        raw_dir,
        seg_dir,
        csv_dir,
        load_data_button,
        raw_image_selector,
        seg_image_selector,
        event_selector,
        add_point_mode,
        save_csv_button,
        status_label,
        progress_bar: mw.ProgressBar,
    ) -> List[napari.types.LayerDataTuple]:
        pass

    # Now define the change handlers AFTER the magicgui function
    @change_handler(plugin.load_data_button)
    def _on_load_data_clicked(value):
        nonlocal raw_files, seg_files

        # Get directories
        raw_directory = str(plugin.raw_dir.value)
        seg_directory = str(plugin.seg_dir.value) if plugin.seg_dir.value else None
        csv_directory = str(plugin.csv_dir.value)

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
        plugin.raw_image_selector.choices = [os.path.basename(f) for f in raw_files]
        if seg_files:
            plugin.seg_image_selector.choices = ["None"] + [os.path.basename(f) for f in seg_files]
        else:
            plugin.seg_image_selector.choices = ["None"]

        # Show widgets
        plugin.raw_image_selector.visible = True
        plugin.seg_image_selector.visible = True
        plugin.event_selector.visible = True

        plugin.status_label.value = f"Loaded: {len(raw_files)} raw images"

        # Auto-load first image if available
        if len(raw_files) > 0:
            _load_current_selection()

    def _load_current_selection():
        """Load currently selected raw image, seg image, and CSV"""
        nonlocal current_raw_image, current_seg_image, current_csv_data, current_event_name, clicked_points

        # Get selected files
        raw_selected = plugin.raw_image_selector.value
        seg_selected = plugin.seg_image_selector.value
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

        # Load CSV for this image and event
        raw_basename = os.path.basename(raw_path)
        image_name = os.path.splitext(raw_basename)[0]
        csv_directory = str(plugin.csv_dir.value)

        # Look for matching CSV file
        csv_pattern = f"{image_name}_oneat_{selected_event}.csv"
        csv_path = os.path.join(csv_directory, csv_pattern)

        if os.path.exists(csv_path):
            current_csv_data = load_csv_file(csv_path, selected_event)
        else:
            # Create empty dataframe if no matching CSV
            current_csv_data = pd.DataFrame(columns=['t', 'z', 'y', 'x'])

        current_event_name = selected_event
        clicked_points = []

        # Clear existing layers
        plugin.viewer.value.layers.clear()

        # Add raw image
        plugin.viewer.value.add_image(current_raw_image, name="Raw Image", colormap="gray")

        # Add seg image if available
        if current_seg_image is not None:
            plugin.viewer.value.add_labels(current_seg_image, name="Segmentation")

        # Add points from CSV
        if len(current_csv_data) > 0:
            points_array = current_csv_data[['t', 'z', 'y', 'x']].values
            plugin.viewer.value.add_points(
                points_array,
                name=f"{selected_event} Events",
                face_color="red",
                edge_color="white",
                size=5,
            )

        # Show annotation controls
        plugin.add_point_mode.visible = True
        plugin.save_csv_button.visible = True

        csv_status = "existing CSV" if os.path.exists(csv_path) else "new CSV (will be created on save)"
        plugin.status_label.value = f"Loaded {image_name} - {selected_event} ({len(current_csv_data)} events, {csv_status})"

        # Register mouse callback if not already registered
        if not hasattr(plugin, '_mouse_callback_registered'):
            @plugin.viewer.value.mouse_double_click_callbacks.append
            def get_event(viewer, event):
                """Handle double-click to add points"""
                nonlocal clicked_points

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
                    plugin.status_label.value = "Error: Unsupported image dimensionality"
                    return

                # Add to clicked points
                new_point = {'t': t, 'z': z, 'y': y, 'x': x}
                clicked_points.append(new_point)

                print(f"Added point: t={t}, z={z}, y={y}, x={x}")
                plugin.status_label.value = f"Added point at t={t}, z={z}, y={y}, x={x} ({len(clicked_points)} new points)"

                # Update points layer
                if len(clicked_points) > 0:
                    # Combine existing CSV data with new points
                    if len(current_csv_data) > 0:
                        combined_df = pd.concat([
                            current_csv_data,
                            pd.DataFrame(clicked_points)
                        ], ignore_index=True)
                    else:
                        combined_df = pd.DataFrame(clicked_points)

                    # Remove old points layer
                    for layer in plugin.viewer.value.layers:
                        if isinstance(layer, napari.layers.Points):
                            plugin.viewer.value.layers.remove(layer)

                    # Add updated points
                    points_array = combined_df[['t', 'z', 'y', 'x']].values
                    plugin.viewer.value.add_points(
                        points_array,
                        name=f"{current_event_name} Events (Updated)",
                        face_color="red",
                        edge_color="white",
                        size=5,
                    )

            plugin._mouse_callback_registered = True

    @change_handler(plugin.raw_image_selector, plugin.seg_image_selector, plugin.event_selector)
    def _on_selection_changed(value):
        """Auto-load when raw image or event type changes"""
        if len(raw_files) > 0 and plugin.raw_image_selector.visible:
            _load_current_selection()

    @change_handler(plugin.save_csv_button)
    def _on_save_csv_clicked(value):
        """Save updated CSV with new points"""
        nonlocal clicked_points

        if len(clicked_points) == 0:
            plugin.status_label.value = "No new points to save"
            return

        # Get current image name
        raw_idx = plugin.raw_image_selector.current_index
        raw_path = raw_files[raw_idx]
        raw_basename = os.path.basename(raw_path)
        image_name = os.path.splitext(raw_basename)[0]

        # Create output CSV filename
        csv_directory = str(plugin.csv_dir.value)
        output_csv = os.path.join(
            csv_directory,
            f"{image_name}_oneat_{current_event_name}.csv"
        )

        # Combine existing and new points
        if len(current_csv_data) > 0:
            combined_df = pd.concat([
                current_csv_data,
                pd.DataFrame(clicked_points)
            ], ignore_index=True)
        else:
            combined_df = pd.DataFrame(clicked_points)

        # Sort by time
        combined_df = combined_df.sort_values('t').reset_index(drop=True)

        # Save to CSV
        combined_df.to_csv(output_csv, index=False)

        plugin.status_label.value = f"Saved {len(combined_df)} points to {os.path.basename(output_csv)}"
        print(f"Saved CSV: {output_csv}")

        # Reset clicked points
        clicked_points = []

    return plugin


# Create the widget
oneat_visualizer_widget = plugin_wrapper_oneat_visualizer()

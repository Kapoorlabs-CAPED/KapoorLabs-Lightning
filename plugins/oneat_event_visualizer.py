"""
ONEAT Event Visualizer and Annotator
Napari plugin for visualizing and adding event annotations
Made by KapoorLabs, 2024
"""

import os
from pathlib import Path
from glob import glob

import napari
import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QComboBox,
    QPushButton,
    QCheckBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
)
from tifffile import imread


class OneatEventVisualizerWidget(QWidget):
    """Widget for ONEAT event visualization and annotation"""

    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent=parent)

        self.viewer = viewer
        self.current_raw_image = None
        self.current_seg_image = None
        self.current_csv_data = None
        self.current_event_name = None
        self.clicked_points = []
        self.raw_files = []
        self.seg_files = []
        self.raw_dir = None
        self.seg_dir = None
        self.csv_dir = None

        self._setup_ui()
        self._connect_callbacks()

    def _setup_ui(self):
        """Setup the UI layout"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Logo and title
        title = QLabel("<h3>ONEAT Event Visualizer & Annotator</h3>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Form layout for inputs
        form_layout = QFormLayout()

        # Directory selection buttons
        self.raw_dir_button = QPushButton("Select Raw Images Directory")
        self.seg_dir_button = QPushButton("Select Seg Images Directory (Optional)")
        self.csv_dir_button = QPushButton("Select CSV Files Directory")

        form_layout.addRow(self.raw_dir_button)
        form_layout.addRow(self.seg_dir_button)
        form_layout.addRow(self.csv_dir_button)

        # Load data button
        self.load_data_button = QPushButton("Load Data")
        self.load_data_button.setEnabled(False)
        form_layout.addRow(self.load_data_button)

        # Image and event selectors
        self.raw_image_box = QComboBox()
        self.raw_image_box.setVisible(False)

        self.seg_image_box = QComboBox()
        self.seg_image_box.setVisible(False)

        self.event_box = QComboBox()
        self.event_box.addItem("mitosis")
        self.event_box.addItem("normal")
        self.event_box.setVisible(False)

        form_layout.addRow("Raw Image:", self.raw_image_box)
        form_layout.addRow("Seg Image:", self.seg_image_box)
        form_layout.addRow("Event Type:", self.event_box)

        # Add point mode checkbox
        self.add_point_checkbox = QCheckBox("Add Point Mode (Double-click)")
        self.add_point_checkbox.setVisible(False)
        form_layout.addRow(self.add_point_checkbox)

        # Save button
        self.save_button = QPushButton("Save Updated CSV")
        self.save_button.setVisible(False)
        form_layout.addRow(self.save_button)

        # Status label
        self.status_label = QLabel("")
        form_layout.addRow("Status:", self.status_label)

        layout.addLayout(form_layout)

        # Table widget for showing points
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(['T', 'Z', 'Y', 'X'])
        self.table_widget.setVisible(False)
        layout.addWidget(self.table_widget)

    def _connect_callbacks(self):
        """Connect all callbacks"""
        self.raw_dir_button.clicked.connect(self._select_raw_dir)
        self.seg_dir_button.clicked.connect(self._select_seg_dir)
        self.csv_dir_button.clicked.connect(self._select_csv_dir)
        self.load_data_button.clicked.connect(self._load_data)
        self.raw_image_box.currentIndexChanged.connect(self._load_current_selection)
        self.seg_image_box.currentIndexChanged.connect(self._load_current_selection)
        self.event_box.currentIndexChanged.connect(self._load_current_selection)
        self.save_button.clicked.connect(self._save_csv)
        self.viewer.mouse_double_click_callbacks.append(self._on_double_click)

    def _select_raw_dir(self):
        """Select raw images directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Raw Images Directory")
        if directory:
            self.raw_dir = directory
            self.status_label.setText(f"Raw dir: {os.path.basename(directory)}")
            self._check_ready_to_load()

    def _select_seg_dir(self):
        """Select seg images directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Seg Images Directory")
        if directory:
            self.seg_dir = directory
            self.status_label.setText(f"Seg dir: {os.path.basename(directory)}")

    def _select_csv_dir(self):
        """Select CSV files directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select CSV Files Directory")
        if directory:
            self.csv_dir = directory
            self.status_label.setText(f"CSV dir: {os.path.basename(directory)}")
            self._check_ready_to_load()

    def _check_ready_to_load(self):
        """Check if we have required directories to enable load button"""
        if self.raw_dir and self.csv_dir:
            self.load_data_button.setEnabled(True)

    def _load_data(self):
        """Load file lists from directories"""
        # Load raw files
        self.raw_files = sorted(glob(os.path.join(self.raw_dir, "*.tif")))

        # Load seg files
        if self.seg_dir:
            self.seg_files = sorted(glob(os.path.join(self.seg_dir, "*.tif")))
        else:
            self.seg_files = []

        if len(self.raw_files) == 0:
            self.status_label.setText("Error: No .tif files in raw directory")
            return

        # Populate raw image dropdown
        self.raw_image_box.clear()
        for f in self.raw_files:
            self.raw_image_box.addItem(os.path.basename(f))

        # Populate seg image dropdown
        self.seg_image_box.clear()
        self.seg_image_box.addItem("None")
        for f in self.seg_files:
            self.seg_image_box.addItem(os.path.basename(f))

        # Show widgets
        self.raw_image_box.setVisible(True)
        self.seg_image_box.setVisible(True)
        self.event_box.setVisible(True)
        self.table_widget.setVisible(True)

        self.status_label.setText(f"Loaded {len(self.raw_files)} raw images")

        # Load first image
        self._load_current_selection()

    def _load_current_selection(self):
        """Load currently selected raw image, seg image, and CSV"""
        raw_idx = self.raw_image_box.currentIndex()
        seg_idx = self.seg_image_box.currentIndex()
        event_name = self.event_box.currentText()

        if raw_idx < 0 or raw_idx >= len(self.raw_files):
            return

        # Load raw image
        raw_path = self.raw_files[raw_idx]
        self.current_raw_image = imread(raw_path)

        # Load seg image if selected
        self.current_seg_image = None
        if seg_idx > 0 and seg_idx <= len(self.seg_files):
            seg_path = self.seg_files[seg_idx - 1]
            self.current_seg_image = imread(seg_path)

        # Load CSV
        raw_basename = os.path.basename(raw_path)
        image_name = os.path.splitext(raw_basename)[0]
        csv_pattern = f"oneat_{event_name}_{image_name}.csv"
        csv_path = os.path.join(self.csv_dir, csv_pattern)

        print(f"Looking for CSV: {csv_path}")

        if os.path.exists(csv_path):
            self.current_csv_data = pd.read_csv(csv_path)
            # Rename time column if needed
            if 'time' in self.current_csv_data.columns:
                self.current_csv_data = self.current_csv_data.rename(columns={'time': 't'})
            print(f"Loaded CSV with {len(self.current_csv_data)} events")
        else:
            self.current_csv_data = pd.DataFrame(columns=['t', 'z', 'y', 'x'])
            print(f"No CSV found, created empty DataFrame")

        self.current_event_name = event_name
        self.clicked_points = []

        # Update viewer
        self._update_viewer()

        # Update table
        self._update_table()

        # Show controls
        self.add_point_checkbox.setVisible(True)
        self.save_button.setVisible(True)

        csv_status = "existing" if os.path.exists(csv_path) else "new"
        self.status_label.setText(
            f"{image_name} - {event_name} ({len(self.current_csv_data)} events, {csv_status})"
        )

    def _update_viewer(self):
        """Update napari viewer with current images and points"""
        # Clear layers
        self.viewer.layers.clear()

        # Add raw image
        self.viewer.add_image(self.current_raw_image, name="Raw Image", colormap="gray")

        # Add seg image if available
        if self.current_seg_image is not None:
            self.viewer.add_labels(self.current_seg_image, name="Segmentation")

        # Add points
        if len(self.current_csv_data) > 0:
            points_array = self.current_csv_data[['t', 'z', 'y', 'x']].values
            print(f"Adding {len(points_array)} points")
            print(f"Image shape: {self.current_raw_image.shape}")

            self.viewer.add_points(
                points_array,
                name=f"{self.current_event_name} Events",
                face_color="red",
                border_color="white",
                size=10,
                ndim=4,
            )

    def _update_table(self):
        """Update table widget with current CSV data"""
        combined_df = self.current_csv_data.copy()

        # Add new clicked points
        if len(self.clicked_points) > 0:
            combined_df = pd.concat([
                combined_df,
                pd.DataFrame(self.clicked_points)
            ], ignore_index=True)

        self.table_widget.setRowCount(len(combined_df))

        for row_idx, (_, row) in enumerate(combined_df.iterrows()):
            self.table_widget.setItem(row_idx, 0, QTableWidgetItem(str(int(row['t']))))
            self.table_widget.setItem(row_idx, 1, QTableWidgetItem(str(int(row['z']))))
            self.table_widget.setItem(row_idx, 2, QTableWidgetItem(str(int(row['y']))))
            self.table_widget.setItem(row_idx, 3, QTableWidgetItem(str(int(row['x']))))

    def _on_double_click(self, viewer, event):
        """Handle double-click to add points"""
        if not self.add_point_checkbox.isChecked():
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

        # Add to clicked points
        new_point = {'t': t, 'z': z, 'y': y, 'x': x}
        self.clicked_points.append(new_point)

        print(f"Added point: t={t}, z={z}, y={y}, x={x}")

        # Update viewer and table
        self._update_viewer_with_new_points()
        self._update_table()

        self.status_label.setText(f"Added point at t={t}, z={z}, y={y}, x={x} ({len(self.clicked_points)} new)")

    def _update_viewer_with_new_points(self):
        """Update points layer with new clicked points"""
        # Combine existing and new points
        combined_df = self.current_csv_data.copy()
        if len(self.clicked_points) > 0:
            combined_df = pd.concat([
                combined_df,
                pd.DataFrame(self.clicked_points)
            ], ignore_index=True)

        # Remove old points layer
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Points):
                self.viewer.layers.remove(layer)

        # Add updated points
        if len(combined_df) > 0:
            points_array = combined_df[['t', 'z', 'y', 'x']].values
            self.viewer.add_points(
                points_array,
                name=f"{self.current_event_name} Events (Updated)",
                face_color="red",
                border_color="white",
                size=10,
                ndim=4,
            )

    def _save_csv(self):
        """Save updated CSV with new points"""
        if len(self.clicked_points) == 0:
            self.status_label.setText("No new points to save")
            return

        # Get current image name
        raw_idx = self.raw_image_box.currentIndex()
        raw_path = self.raw_files[raw_idx]
        raw_basename = os.path.basename(raw_path)
        image_name = os.path.splitext(raw_basename)[0]

        # Create output CSV filename
        output_csv = os.path.join(
            self.csv_dir,
            f"oneat_{self.current_event_name}_{image_name}.csv"
        )

        # Combine existing and new points
        combined_df = self.current_csv_data.copy()
        if len(self.clicked_points) > 0:
            combined_df = pd.concat([
                combined_df,
                pd.DataFrame(self.clicked_points)
            ], ignore_index=True)

        # Sort by time
        combined_df = combined_df.sort_values('t').reset_index(drop=True)

        # Save to CSV
        combined_df.to_csv(output_csv, index=False)

        self.status_label.setText(f"Saved {len(combined_df)} points to {os.path.basename(output_csv)}")
        print(f"Saved CSV: {output_csv}")

        # Update current data and reset clicked points
        self.current_csv_data = combined_df.copy()
        self.clicked_points = []

        # Update table
        self._update_table()


def oneat_visualizer_widget():
    """Factory function for napari plugin"""
    def create_widget(viewer):
        return OneatEventVisualizerWidget(viewer)
    return create_widget

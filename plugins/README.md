# ONEAT Event Visualizer - Napari Plugin

Napari plugin for visualizing and annotating cell events in microscopy timelapses.

## Features

- **CSV Visualizer**: View existing event annotations on microscopy images
- **Training Data Maker**: Add new event annotations interactively
- **Multi-event Support**: Handle multiple event types (mitosis, apoptosis, etc.)
- **Segmentation Overlay**: Optionally display segmentation masks

## Installation

```bash
# From plugins directory
pip install -e .
```

Or run directly:
```bash
napari -w oneat-event-visualizer
```

## Usage

### 1. Load Data

1. Select **Raw Images Directory**: Folder containing `.tif` timelapse files
2. Select **Seg Images Directory** (optional): Folder with segmentation masks
3. Select **CSV Files Directory**: Folder with event annotation CSVs
4. Click **Load Data**

### 2. Select Image and Event

1. Choose raw image from dropdown
2. Optionally choose corresponding segmentation
3. Select event type (extracted from CSV filenames)
4. Click **Load Images & Events**

### 3. Visualize and Annotate

- Existing events from CSV displayed as red points
- Enable **Add Point Mode** checkbox
- **Double-click** on image to add new event points
- Points added at current timepoint and location

### 4. Save Annotations

Click **Save Updated CSV** to save combined existing + new annotations

Output format:
```csv
t,z,y,x
10,5,128,256
15,6,130,258
```

## CSV File Naming Convention

Expected format: `{image_name}_oneat_{event_name}.csv`

Examples:
- `movie01_oneat_mitosis.csv`
- `movie01_oneat_apoptosis.csv`

## Keyboard Shortcuts

- **Double-click**: Add point (when Add Point Mode enabled)
- **Mouse scroll**: Navigate through time/Z-stack
- **Shift + Mouse**: Pan view
- **Ctrl + Mouse**: Zoom

## Workflow Integration

1. **Before Training**: Use to create/augment training annotations
2. **After Training**: Use to visualize predictions alongside ground truth
3. **Quality Control**: Review and correct annotations

## File Structure

```
plugins/
├── oneat_event_visualizer.py  # Main plugin code
├── napari.yaml                 # Napari plugin manifest
├── __init__.py                 # Package init
├── resources/
│   └── kapoorlogo.png         # Logo
└── README.md                   # This file
```

## Tips

- Use descriptive event names in CSV filenames
- Keep raw and seg images with matching names
- Sort files chronologically for easier navigation
- Save frequently when adding many annotations

## Troubleshooting

**No images loading**: Check .tif file extensions and directory paths

**Points not appearing**: Verify CSV has columns: t, z, y, x

**Can't add points**: Enable "Add Point Mode" checkbox

**Wrong event displayed**: Check CSV filename matches event selector

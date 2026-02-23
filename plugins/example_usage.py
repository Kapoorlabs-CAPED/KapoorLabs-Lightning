"""
Example usage of ONEAT Event Visualizer plugin
"""

import napari
from oneat_event_visualizer import OneatEventVisualizerWidget

if __name__ == "__main__":
    # Create napari viewer
    viewer = napari.Viewer()

    # Add the ONEAT visualizer widget
    widget = OneatEventVisualizerWidget(viewer)
    viewer.window.add_dock_widget(widget, name="ONEAT Events")

    # Start napari
    napari.run()

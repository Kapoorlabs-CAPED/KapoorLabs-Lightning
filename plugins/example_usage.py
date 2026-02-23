"""
Example usage of ONEAT Event Visualizer plugin
"""

import napari
from oneat_event_visualizer import oneat_visualizer_widget

if __name__ == "__main__":
    # Create napari viewer
    viewer = napari.Viewer()

    # Add the ONEAT visualizer widget
    viewer.window.add_dock_widget(oneat_visualizer_widget, name="ONEAT Events")

    # Start napari
    napari.run()

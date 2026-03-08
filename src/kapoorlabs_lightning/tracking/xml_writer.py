"""
TrackMate XML writer.

Writes modified TrackMate XML files — either enhanced master XMLs with
computed properties or channel-transferred XMLs.
"""

import os
import lxml.etree as et
from typing import Dict, Any, Optional
from pathlib import Path

from .xml_parser import TrackMateXML


def write_trackmate_xml(
    parsed_xml: TrackMateXML,
    spot_properties: Dict[int, Dict[str, Any]],
    output_path: str,
    output_name: Optional[str] = None,
):
    """
    Write a modified TrackMate XML with updated spot properties.

    Takes the original XML tree and updates each spot's attributes
    with values from spot_properties, then writes to disk.

    Args:
        parsed_xml: Parsed TrackMateXML object.
        spot_properties: Dict mapping cell_id → property dict.
            Keys should match XML attribute names.
        output_path: Directory to write the output XML.
        output_name: Output filename. Defaults to 'master_<original>.xml'.
    """
    xml_tree = et.parse(parsed_xml.xml_path)
    xml_root = xml_tree.getroot()

    spot_objects = xml_root.find("Model").find("AllSpots")

    for frame_node in spot_objects.findall("SpotsInFrame"):
        for spot_node in frame_node.findall("Spot"):
            cell_id = int(spot_node.get("ID"))

            if cell_id in spot_properties:
                props = spot_properties[cell_id]
                for key, value in props.items():
                    spot_node.set(key, str(value))

    if output_name is None:
        base = os.path.splitext(os.path.basename(parsed_xml.xml_path))[0]
        output_name = f"master_{base}.xml"

    Path(output_path).mkdir(parents=True, exist_ok=True)
    xml_tree.write(os.path.join(output_path, output_name))


def write_channel_xml(
    parsed_xml: TrackMateXML,
    channel_spot_properties: Dict[int, Dict[str, Any]],
    output_path: str,
    channel_name: str = "membrane",
):
    """
    Write a channel-transferred TrackMate XML.

    Creates a new XML file with spot positions and properties
    transferred from the original channel to the target channel.

    Args:
        parsed_xml: Parsed TrackMateXML from the source channel.
        channel_spot_properties: Dict mapping cell_id → transferred
            properties (positions, intensities, etc.).
        output_path: Directory to write the output XML.
        channel_name: Name of the target channel (used in filename).
    """
    xml_tree = et.parse(parsed_xml.xml_path)
    xml_root = xml_tree.getroot()

    spot_objects = xml_root.find("Model").find("AllSpots")

    for frame_node in spot_objects.findall("SpotsInFrame"):
        for spot_node in frame_node.findall("Spot"):
            cell_id = int(spot_node.get("ID"))

            if cell_id in channel_spot_properties:
                props = channel_spot_properties[cell_id]

                # Update position
                if "POSITION_Z" in props:
                    spot_node.set("POSITION_Z", str(props["POSITION_Z"]))
                if "POSITION_Y" in props:
                    spot_node.set("POSITION_Y", str(props["POSITION_Y"]))
                if "POSITION_X" in props:
                    spot_node.set("POSITION_X", str(props["POSITION_X"]))

                # Update intensity
                for key in [
                    "MEAN_INTENSITY_CH1",
                    "MEAN_INTENSITY_CH2",
                    "TOTAL_INTENSITY_CH1",
                    "TOTAL_INTENSITY_CH2",
                    "RADIUS",
                    "QUALITY",
                ]:
                    if key in props:
                        spot_node.set(key, str(props[key]))

    # Build output filename
    base = os.path.splitext(os.path.basename(parsed_xml.xml_path))[0]
    if "nuclei" in base:
        output_name = base.replace("nuclei", channel_name) + ".xml"
    else:
        output_name = f"{base}_{channel_name}.xml"

    Path(output_path).mkdir(parents=True, exist_ok=True)
    xml_tree.write(os.path.join(output_path, output_name))

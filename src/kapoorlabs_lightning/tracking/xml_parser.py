"""
TrackMate XML parser.

Reads TrackMate v6.0+ XML files and extracts spots, edges, tracks,
calibration data, and filtered track IDs into clean data structures.
"""

import numpy as np
import lxml.etree as et
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class SpotData:
    """Properties of a single detected spot."""

    cell_id: int
    frame: int
    z: float
    y: float
    x: float
    radius: float
    quality: float
    mean_intensity: float = -1.0
    total_intensity: float = -1.0


@dataclass
class Calibration:
    """Image calibration from TrackMate XML."""

    x: float = 1.0
    y: float = 1.0
    z: float = 1.0
    t: float = 1.0
    nframes: int = 0
    nslices: int = 0
    height: int = 0
    width: int = 0


@dataclass
class TrackData:
    """Properties of a single track from the XML."""

    track_id: int
    displacement: float = 0.0
    total_distance: float = 0.0
    max_distance: float = 0.0
    duration: float = 0.0
    num_splits: int = 0
    source_ids: List[int] = field(default_factory=list)
    target_ids: List[int] = field(default_factory=list)


class TrackMateXML:
    """
    Parser for TrackMate XML files.

    Reads the XML and provides clean access to spots, edges, tracks,
    calibration, and lineage structure.

    Args:
        xml_path: Path to TrackMate XML file.

    Attributes:
        spots: Dict mapping cell_id → SpotData.
        calibration: Calibration dataclass.
        filtered_track_ids: List of filtered track IDs.
        edge_target_lookup: Dict mapping source_id → [target_ids].
        edge_source_lookup: Dict mapping target_id → source_id.
        tracks: Dict mapping track_id → TrackData.
    """

    def __init__(self, xml_path: str):
        self.xml_path = str(xml_path)
        self.spots: Dict[int, SpotData] = {}
        self.calibration = Calibration()
        self.filtered_track_ids: List[int] = []
        self.edge_target_lookup: Dict[int, List[int]] = {}
        self.edge_source_lookup: Dict[int, int] = {}
        self.tracks: Dict[int, TrackData] = {}
        self.detector_channel: int = 1

        self._xml_content = None
        self._xml_tree = None

        self._parse()

    def _parse(self):
        """Parse the XML file."""
        parser = et.XMLParser(huge_tree=True)
        self._xml_content = et.fromstring(
            open(self.xml_path).read().encode(), parser
        )
        self._xml_tree = et.parse(self.xml_path)

        # Filtered track IDs
        filtered_tracks_node = (
            self._xml_content.find("Model").find("FilteredTracks")
        )
        self.filtered_track_ids = [
            int(track.get("TRACK_ID"))
            for track in filtered_tracks_node.findall("TrackID")
        ]

        # Calibration
        settings = self._xml_content.find("Settings").find("ImageData")
        self.calibration = Calibration(
            x=float(settings.get("pixelwidth")),
            y=float(settings.get("pixelheight")),
            z=float(settings.get("voxeldepth")),
            t=float(settings.get("timeinterval")),
            nframes=int(float(settings.get("nframes"))),
            nslices=int(float(settings.get("nslices"))),
            height=int(float(settings.get("height"))),
            width=int(float(settings.get("width"))),
        )

        # Detector channel
        detector_settings = self._xml_content.find("Settings").find(
            "DetectorSettings"
        )
        try:
            self.detector_channel = int(
                float(detector_settings.get("TARGET_CHANNEL"))
            )
        except (TypeError, ValueError):
            self.detector_channel = 1

        # Parse spots
        self._parse_spots()

        # Parse tracks and edges
        self._parse_tracks()

    def _parse_spots(self):
        """Extract all spots from the XML."""
        spot_objects = self._xml_content.find("Model").find("AllSpots")

        for frame_node in spot_objects.findall("SpotsInFrame"):
            for spot_node in frame_node.findall("Spot"):
                cell_id = int(spot_node.get("ID"))

                # Intensity based on detector channel
                if self.detector_channel == 1:
                    total_key = "TOTAL_INTENSITY_CH2"
                    mean_key = "MEAN_INTENSITY_CH2"
                else:
                    total_key = "TOTAL_INTENSITY_CH1"
                    mean_key = "MEAN_INTENSITY_CH1"

                total_intensity = (
                    float(spot_node.get(total_key))
                    if spot_node.get(total_key) is not None
                    else -1.0
                )
                mean_intensity = (
                    float(spot_node.get(mean_key))
                    if spot_node.get(mean_key) is not None
                    else -1.0
                )

                self.spots[cell_id] = SpotData(
                    cell_id=cell_id,
                    frame=int(spot_node.get("FRAME")),
                    z=float(spot_node.get("POSITION_Z")),
                    y=float(spot_node.get("POSITION_Y")),
                    x=float(spot_node.get("POSITION_X")),
                    radius=float(spot_node.get("RADIUS")),
                    quality=float(spot_node.get("QUALITY")),
                    mean_intensity=mean_intensity,
                    total_intensity=total_intensity,
                )

    def _parse_tracks(self):
        """Extract all tracks and edges from the XML."""
        all_tracks = self._xml_content.find("Model").find("AllTracks")

        for track_node in all_tracks.findall("Track"):
            track_id = int(track_node.get("TRACK_ID"))

            if track_id not in self.filtered_track_ids:
                continue

            # Track-level features
            track_data = TrackData(
                track_id=track_id,
                displacement=float(
                    track_node.get("TRACK_DISPLACEMENT", 0)
                ),
                total_distance=float(
                    track_node.get("TOTAL_DISTANCE_TRAVELED", 0)
                ),
                max_distance=float(
                    track_node.get("MAX_DISTANCE_TRAVELED", 0)
                ),
                duration=float(track_node.get("TRACK_DURATION", 0)),
            )

            # Edges
            source_ids = []
            target_ids = []
            for edge_node in track_node.findall("Edge"):
                source_id = int(edge_node.get("SPOT_SOURCE_ID"))
                target_id = int(edge_node.get("SPOT_TARGET_ID"))

                source_ids.append(source_id)
                target_ids.append(target_id)

                # Build lookup tables
                if source_id in self.edge_target_lookup:
                    self.edge_target_lookup[source_id].append(target_id)
                else:
                    self.edge_target_lookup[source_id] = [target_id]
                self.edge_source_lookup[target_id] = source_id

            track_data.source_ids = source_ids
            track_data.target_ids = target_ids
            track_data.num_splits = self._count_splits(source_ids)
            self.tracks[track_id] = track_data

    def _count_splits(self, source_ids: List[int]) -> int:
        """Count the number of split events in a track."""
        count = 0
        for source_id in source_ids:
            if source_id in self.edge_target_lookup:
                if len(self.edge_target_lookup[source_id]) > 1:
                    count += 1
        return count

    def get_root_ids(self, track_id: int) -> List[int]:
        """Get root spot IDs for a track (spots with no parent)."""
        track = self.tracks[track_id]
        roots = []
        for source_id in track.source_ids:
            if source_id not in self.edge_source_lookup:
                if source_id not in roots:
                    roots.append(source_id)
        return roots

    def get_leaf_ids(self, track_id: int) -> List[int]:
        """Get leaf spot IDs for a track (spots with no children)."""
        track = self.tracks[track_id]
        leaves = []
        all_targets = set()
        for source_id in track.source_ids:
            for target_id in self.edge_target_lookup.get(source_id, []):
                all_targets.add(target_id)

        for target_id in all_targets:
            if target_id not in self.edge_target_lookup:
                leaves.append(target_id)
        return leaves

    def get_split_ids(self, track_id: int) -> List[int]:
        """Get spot IDs where cell divisions occur."""
        track = self.tracks[track_id]
        splits = []
        for source_id in track.source_ids:
            if source_id in self.edge_target_lookup:
                if len(self.edge_target_lookup[source_id]) > 1:
                    if source_id not in splits:
                        splits.append(source_id)
        return splits

    def get_all_spot_ids_for_track(self, track_id: int) -> List[int]:
        """Get all spot IDs belonging to a track."""
        track = self.tracks[track_id]
        all_ids = set(track.source_ids) | set(track.target_ids)
        return sorted(all_ids)

    def is_dividing(self, track_id: int) -> bool:
        """Check if a track has any division events."""
        return self.tracks[track_id].num_splits > 0

    @property
    def xml_tree(self):
        """Access the parsed lxml tree (for XML writing)."""
        return self._xml_tree

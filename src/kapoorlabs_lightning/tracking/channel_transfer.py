"""
Cross-channel track transfer.

Transfers tracks from one channel (e.g. nuclei) to another channel
(e.g. membrane) using spatial matching of segmentation labels.
Produces transferred spot properties and a new XML file.
"""

import numpy as np
import concurrent.futures
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from scipy import spatial
from skimage.measure import regionprops

from .xml_parser import TrackMateXML
from .xml_writer import write_channel_xml


@dataclass
class ChannelRegionData:
    """Pre-computed data for a single timeframe in the target channel."""

    tree: spatial.cKDTree
    centroids: List[Tuple]
    labels: List[int]
    volumes: List[float]
    intensity_means: List[float]
    intensity_totals: List[float]
    bounding_boxes: List[Tuple]


class ChannelTransfer:
    """
    Transfer tracks from a source channel to a target channel.

    Uses spatial matching (KDTree) to find the nearest segmented
    object in the target channel for each spot in the source channel.

    Args:
        parsed_xml: TrackMateXML parsed from the source channel.
        target_seg_image: Labeled segmentation of the target channel.
            Shape: (T, Z, Y, X) or (T, Y, X).
        target_intensity_image: Optional intensity image for the target
            channel. If None, uses target_seg_image for intensity.
        channel_name: Name for the target channel (used in output).
        num_workers: Number of threads for parallel processing.

    Example:
        >>> xml = TrackMateXML("nuclei_tracks.xml")
        >>> transfer = ChannelTransfer(
        ...     xml, membrane_seg, channel_name="membrane"
        ... )
        >>> transferred = transfer.run()
        >>> transfer.write_xml("/output/dir/")
    """

    def __init__(
        self,
        parsed_xml: TrackMateXML,
        target_seg_image: np.ndarray,
        target_intensity_image: Optional[np.ndarray] = None,
        channel_name: str = "membrane",
        num_workers: Optional[int] = None,
    ):
        self.parsed_xml = parsed_xml
        self.target_seg_image = target_seg_image.astype(np.uint16)
        self.target_intensity_image = (
            target_intensity_image
            if target_intensity_image is not None
            else target_seg_image
        )
        self.channel_name = channel_name
        self.num_workers = num_workers or os.cpu_count()

        self._channel_data: Dict[int, ChannelRegionData] = {}
        self.transferred_properties: Dict[int, Dict] = {}
        self.matched_tracks: List[int] = []

    def _build_channel_tree(self):
        """Build KDTree spatial index for each frame of the target channel."""
        futures = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            for t in range(self.target_seg_image.shape[0]):
                futures.append(executor.submit(self._process_frame, t))
            for f in concurrent.futures.as_completed(futures):
                f.result()

    def _process_frame(self, t: int):
        """Extract regionprops and build KDTree for one frame."""
        seg_frame = self.target_seg_image[t]
        intensity_frame = self.target_intensity_image[t]

        properties = regionprops(seg_frame, intensity_image=intensity_frame)

        if not properties:
            return

        centroids = [prop.centroid for prop in properties]
        labels = [prop.label for prop in properties]
        volumes = [prop.area for prop in properties]
        intensity_means = [prop.intensity_mean for prop in properties]
        intensity_totals = [
            prop.intensity_mean * prop.area for prop in properties
        ]
        bounding_boxes = [prop.bbox for prop in properties]

        tree = spatial.cKDTree(centroids)

        self._channel_data[t] = ChannelRegionData(
            tree=tree,
            centroids=centroids,
            labels=labels,
            volumes=volumes,
            intensity_means=intensity_means,
            intensity_totals=intensity_totals,
            bounding_boxes=bounding_boxes,
        )

    def _match_spot(self, cell_id: int) -> bool:
        """
        Match a source spot to the nearest target channel region.

        Returns True if a match was found within the veto radius.
        """
        spot = self.parsed_xml.spots.get(cell_id)
        if spot is None:
            return False

        frame = spot.frame
        if frame not in self._channel_data:
            return False

        channel_data = self._channel_data[frame]
        cal = self.parsed_xml.calibration

        # Source position in pixel coordinates
        source_pos = (
            spot.z / cal.z,
            spot.y / cal.y,
            spot.x / cal.x,
        )

        # Query nearest in target channel
        distance, idx = channel_data.tree.query(source_pos)

        # Compute veto radius from volume
        volume = channel_data.volumes[idx]
        veto_radius = 2 * (3 * volume / (4 * np.pi)) ** (1 / 3)

        if distance <= 2 * veto_radius:
            centroid = channel_data.centroids[idx]
            self.transferred_properties[cell_id] = {
                "POSITION_Z": centroid[0] * cal.z if len(centroid) > 2 else 0,
                "POSITION_Y": centroid[-2] * cal.y,
                "POSITION_X": centroid[-1] * cal.x,
                "MEAN_INTENSITY_CH1": channel_data.intensity_means[idx],
                "TOTAL_INTENSITY_CH1": channel_data.intensity_totals[idx],
                "RADIUS": (volume ** (1 / 3)),
                "QUALITY": channel_data.volumes[idx],
            }
            return True
        return False

    def run(self) -> Dict[int, Dict]:
        """
        Run the channel transfer for all spots in filtered tracks.

        Returns:
            Dict mapping cell_id → transferred properties.
        """
        print(f"Building spatial index for {self.channel_name} channel...")
        self._build_channel_tree()

        print("Matching spots to target channel...")
        for track_id in self.parsed_xml.filtered_track_ids:
            spot_ids = self.parsed_xml.get_all_spot_ids_for_track(track_id)
            track_matched = False
            for cell_id in spot_ids:
                if self._match_spot(cell_id):
                    track_matched = True
            if track_matched:
                self.matched_tracks.append(track_id)

        print(
            f"Matched {len(self.transferred_properties)} spots "
            f"across {len(self.matched_tracks)} tracks"
        )
        return self.transferred_properties

    def write_xml(self, output_path: str):
        """
        Write the channel-transferred XML file.

        Args:
            output_path: Directory to save the output XML.
        """
        if not self.transferred_properties:
            raise ValueError(
                "No transferred properties. Call run() first."
            )

        write_channel_xml(
            self.parsed_xml,
            self.transferred_properties,
            output_path,
            self.channel_name,
        )
        print(f"Channel XML written to {output_path}")

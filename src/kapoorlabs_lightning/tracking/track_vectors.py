"""
Track feature assembly — the main user-facing API for tracking analysis.

Assembles shape, dynamic, and tracking features into DataFrames
from TrackMate XML data with optional segmentation-based morphology.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from scipy import spatial
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

from .xml_parser import TrackMateXML
from .track_features import (
    compute_speed,
    compute_acceleration,
    compute_motion_angles,
    compute_radial_angles,
    compute_msd,
    SHAPE_FEATURES,
    DYNAMIC_FEATURES,
    SHAPE_DYNAMIC_FEATURES,
    IDENTITY_FEATURES,
    ALL_FEATURES,
)


class TrackVectors:
    """
    Main API for assembling track feature vectors from TrackMate XML.

    Combines parsed XML data with optional segmentation-based shape
    features to produce complete feature DataFrames.

    Args:
        xml_path: Path to TrackMate XML file.
        seg_image: Optional labeled segmentation image (T, Z, Y, X)
            for computing shape features via point clouds.
        mask: Optional binary mask image for distance-to-boundary
            computation.
        calibration: Override calibration as (z, y, x) tuple.
            If None, uses calibration from the XML.

    Example:
        >>> tv = TrackVectors("tracks.xml", seg_image=seg)
        >>> df = tv.to_dataframe()
        >>> df.to_csv("track_features.csv")
    """

    def __init__(
        self,
        xml_path: str,
        seg_image: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        calibration: Optional[Tuple[float, float, float]] = None,
    ):
        self.xml = TrackMateXML(xml_path)
        self.seg_image = seg_image
        self.mask = mask

        if calibration is not None:
            self._cal = calibration
        else:
            self._cal = (
                self.xml.calibration.z,
                self.xml.calibration.y,
                self.xml.calibration.x,
            )

        # Precomputed data
        self._boundary_trees = {}
        self._shape_features = {}

        # Compute boundary trees if mask provided
        if self.mask is not None:
            self._build_boundary_trees()

        # Compute shape features if seg image provided
        if self.seg_image is not None:
            self._compute_shape_features()

    def _build_boundary_trees(self):
        """Build KDTree for mask boundary points per timeframe."""
        for t in range(self.mask.shape[0]):
            frame_mask = self.mask[t]
            boundaries = find_boundaries(frame_mask, mode="inner")
            boundary_coords = np.argwhere(boundaries)
            if len(boundary_coords) > 0:
                # Scale to calibrated coordinates
                scaled = boundary_coords.astype(float)
                scaled[:, 0] *= self._cal[0]  # z
                if scaled.shape[1] > 2:
                    scaled[:, 1] *= self._cal[1]  # y
                    scaled[:, 2] *= self._cal[2]  # x
                else:
                    scaled[:, 0] *= self._cal[1]  # y
                    scaled[:, 1] *= self._cal[2]  # x
                centroid = np.mean(scaled, axis=0)
                tree = spatial.cKDTree(scaled)
                self._boundary_trees[t] = (tree, centroid)

    def _compute_shape_features(self):
        """Compute shape features from segmentation using morphology module."""
        from ..morphology.shape_features import compute_shape_features_for_image

        df = compute_shape_features_for_image(
            self.seg_image,
            num_points=2048,
            calibration=self._cal,
        )
        # Index by (t, centroid) for matching to spots
        for _, row in df.iterrows():
            t = int(row.get("t", 0))
            centroid = (
                row.get("centroid_z", 0),
                row.get("centroid_y", 0),
                row.get("centroid_x", 0),
            )
            self._shape_features[(t, row["label"])] = row

    def _get_boundary_distance(
        self, frame: int, position: Tuple[float, float, float]
    ) -> Tuple[float, Tuple]:
        """Get distance from position to nearest mask boundary."""
        if frame not in self._boundary_trees:
            return 0.0, (0, 0, 0)
        tree, centroid = self._boundary_trees[frame]
        dist, _ = tree.query(position)
        return max(0.0, dist), tuple(centroid)

    def _get_local_density(
        self, frame: int, position: Tuple[float, float, float], radius: float = 50.0
    ) -> int:
        """Count unique labels in neighborhood around position."""
        if self.seg_image is None:
            return 0
        frame_seg = self.seg_image[frame]
        z, y, x = [int(p / c) for p, c in zip(position, self._cal)]

        if frame_seg.ndim == 3:
            r = int(radius)
            z0, z1 = max(0, z - r), min(frame_seg.shape[0], z + r)
            y0, y1 = max(0, y - r), min(frame_seg.shape[1], y + r)
            x0, x1 = max(0, x - r), min(frame_seg.shape[2], x + r)
            region = frame_seg[z0:z1, y0:y1, x0:x1]
        else:
            r = int(radius)
            y0, y1 = max(0, y - r), min(frame_seg.shape[0], y + r)
            x0, x1 = max(0, x - r), min(frame_seg.shape[1], x + r)
            region = frame_seg[y0:y1, x0:x1]

        return len(np.unique(region))

    def _match_shape_to_spot(
        self, frame: int, position: Tuple[float, float, float]
    ) -> Optional[pd.Series]:
        """Find the closest shape feature entry for a spot."""
        if not self._shape_features:
            return None

        # Find closest centroid at this frame
        frame_shapes = {
            k: v for k, v in self._shape_features.items() if k[0] == frame
        }
        if not frame_shapes:
            return None

        pos_pixel = np.array([p / c for p, c in zip(position, self._cal)])
        best_dist = float("inf")
        best_row = None

        for (t, label), row in frame_shapes.items():
            centroid = np.array([
                row.get("centroid_z", 0),
                row.get("centroid_y", 0),
                row.get("centroid_x", 0),
            ])
            dist = np.linalg.norm(pos_pixel - centroid)
            if dist < best_dist:
                best_dist = dist
                best_row = row

        return best_row

    def to_dataframe(self) -> pd.DataFrame:
        """
        Assemble all features into a DataFrame.

        Returns:
            DataFrame with columns for identity, shape, dynamic,
            and tracking features for every spot in every filtered track.
        """
        rows = []

        for track_id in self.xml.filtered_track_ids:
            track = self.xml.tracks.get(track_id)
            if track is None:
                continue

            spot_ids = self.xml.get_all_spot_ids_for_track(track_id)

            # Sort spots by time
            spot_ids_sorted = sorted(
                spot_ids,
                key=lambda sid: self.xml.spots[sid].frame,
            )

            is_dividing = self.xml.is_dividing(track_id)
            num_splits = track.num_splits

            # Collect positions for MSD
            positions = []
            for sid in spot_ids_sorted:
                spot = self.xml.spots[sid]
                positions.append([spot.z, spot.y, spot.x])
            positions = np.array(positions)
            track_msd = compute_msd(positions)

            prev_speed = 0.0
            prev_pos = None

            for i, sid in enumerate(spot_ids_sorted):
                spot = self.xml.spots[sid]
                pos = (spot.z, spot.y, spot.x)

                # Speed
                if prev_pos is not None:
                    speed = compute_speed(pos, prev_pos, self._cal)
                else:
                    speed = 0.0

                # Acceleration
                accel = compute_acceleration(speed, prev_speed)

                # Motion angles
                if prev_pos is not None:
                    m_z, m_y, m_x = compute_motion_angles(pos, prev_pos)
                else:
                    m_z, m_y, m_x = 0.0, 0.0, 0.0

                # Radial angles
                r_z, r_y, r_x = compute_radial_angles(np.array(pos))

                # Distance to mask boundary
                dist_mask, mask_centroid = self._get_boundary_distance(
                    spot.frame, pos
                )

                # Local cell density
                local_density = self._get_local_density(spot.frame, pos)

                # Shape features from segmentation
                shape_row = self._match_shape_to_spot(spot.frame, pos)
                if shape_row is not None:
                    ecc1 = shape_row.get("eccentricity_comp_1", np.nan)
                    ecc2 = shape_row.get("eccentricity_comp_2", np.nan)
                    ecc3 = shape_row.get("eccentricity_comp_3", np.nan)
                    surf_area = shape_row.get("surface_area", np.nan)
                    axis_z = shape_row.get("cell_axis_z", np.nan)
                    axis_y = shape_row.get("cell_axis_y", np.nan)
                    axis_x = shape_row.get("cell_axis_x", np.nan)
                else:
                    ecc1 = ecc2 = ecc3 = np.nan
                    surf_area = np.nan
                    axis_z = axis_y = axis_x = np.nan

                row = {
                    "Track_ID": track_id,
                    "t": spot.frame,
                    "z": spot.z,
                    "y": spot.y,
                    "x": spot.x,
                    "Dividing": int(is_dividing),
                    "Number_Dividing": num_splits,
                    "Radius": spot.radius,
                    "Eccentricity_Comp_First": ecc1,
                    "Eccentricity_Comp_Second": ecc2,
                    "Eccentricity_Comp_Third": ecc3,
                    "Local_Cell_Density": local_density,
                    "Surface_Area": surf_area,
                    "Speed": speed,
                    "Motion_Angle_Z": m_z,
                    "Motion_Angle_Y": m_y,
                    "Motion_Angle_X": m_x,
                    "Acceleration": accel,
                    "Distance_Cell_mask": dist_mask,
                    "Radial_Angle_Z": r_z,
                    "Radial_Angle_Y": r_y,
                    "Radial_Angle_X": r_x,
                    "Cell_Axis_Z": axis_z,
                    "Cell_Axis_Y": axis_y,
                    "Cell_Axis_X": axis_x,
                    "MSD": track_msd,
                    "Track_Displacement": track.displacement,
                    "Total_Track_Distance": track.total_distance,
                    "Max_Track_Distance": track.max_distance,
                    "Track_Duration": track.duration,
                    "Total_Intensity": spot.total_intensity,
                    "Mean_Intensity": spot.mean_intensity,
                }
                rows.append(row)

                prev_speed = speed
                prev_pos = pos

        return pd.DataFrame(rows)

    def get_shape_features_only(self) -> pd.DataFrame:
        """Get only shape features (no tracking required)."""
        if self.seg_image is None:
            raise ValueError("seg_image is required for shape features")

        from ..morphology.shape_features import compute_shape_features_for_image

        return compute_shape_features_for_image(
            self.seg_image,
            num_points=2048,
            calibration=self._cal,
        )

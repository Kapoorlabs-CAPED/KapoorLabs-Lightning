"""
Track feature assembly — the main user-facing API for tracking analysis.

Assembles shape, dynamic, and tracking features into DataFrames
from TrackMate XML data with optional segmentation-based morphology.
"""

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from scipy import spatial
from skimage.segmentation import find_boundaries

from ..morphology.point_clouds import extract_all_point_clouds
from ..morphology.shape_features import (
    compute_shape_features as compute_single_shape_features,
    compute_shape_features_for_image,
)
from .xml_parser import TrackMateXML
from .track_features import (
    compute_speed,
    compute_acceleration,
    compute_dt,
    compute_motion_angles,
    compute_radial_angles,
    compute_msd,
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
        variable_t_calibration: Optional[Dict[int, float]] = None,
        is_master_xml: Optional[bool] = None,
    ):
        self.xml = TrackMateXML(xml_path)
        self.seg_image = seg_image
        self.mask = mask
        self._variable_t_calibration = (
            dict(variable_t_calibration) if variable_t_calibration else None
        )
        # Resolve master-XML mode:
        #   None  -> auto-detect from XML contents
        #   True  -> force master mode (fail loudly if attrs missing)
        #   False -> recompute even if XML carries precomputed features
        if is_master_xml is None:
            self.is_master_xml = self.xml.is_master
        else:
            self.is_master_xml = bool(is_master_xml)
        if self.is_master_xml and not self.xml.is_master:
            raise ValueError(
                "is_master_xml=True but XML has no master attributes "
                "(e.g. 'unique_id' on <Spot>). Provide a NapaTrackMater master XML."
            )

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
        # Maps spot_id → ShapeFeatures dict for spots matched via KDTree
        self._spot_shape_features: Dict[int, Dict] = {}

        # Compute boundary trees if mask provided
        if self.mask is not None:
            self._build_boundary_trees()

        # Build per-timeframe KDTree of XML spot centroids
        self._timed_spot_trees: Dict[
            int, Tuple[spatial.cKDTree, np.ndarray, List[int]]
        ] = {}
        self._build_spot_trees()

        # Compute shape features and match to spots via KDTree
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

    def _build_spot_trees(self):
        """Build per-timeframe KDTree from XML spot centroids.

        For each timeframe, collects all spot positions and builds a
        KDTree so that point cloud centroids can be matched to the
        nearest XML spot (same approach as NapaTrackMater).
        """
        timed_spots = defaultdict(list)
        for spot_id, spot in self.xml.spots.items():
            timed_spots[spot.frame].append((spot_id, spot))

        for frame, spots_in_frame in timed_spots.items():
            spot_ids = [s[0] for s in spots_in_frame]
            centroids = np.array([[s[1].z, s[1].y, s[1].x] for s in spots_in_frame])
            tree = spatial.cKDTree(centroids)
            self._timed_spot_trees[frame] = (tree, centroids, spot_ids)

    def _compute_shape_features(self):
        """Compute shape features from segmentation and match to spots via KDTree.

        For each timeframe:
          1. Extract point clouds from the segmentation.
          2. Compute shape features (eccentricity, surface area, cell axis).
          3. For each point cloud centroid, query the per-frame KDTree of
             XML spot centroids to find the nearest spot.
          4. Only assign if distance < quality (geometric mean of eigenvalues),
             matching the NapaTrackMater approach.
        """
        ndim = self.seg_image.ndim
        is_timelapse = ndim == 4 or (
            ndim == 3 and self.seg_image.shape[0] < self.seg_image.shape[-1]
        )

        if ndim <= 3 and not is_timelapse:
            frames = [(0, self.seg_image)]
        else:
            frames = [(t, self.seg_image[t]) for t in range(self.seg_image.shape[0])]

        min_size = (
            (2, 2, 2) if (ndim >= 3 and not (ndim == 3 and is_timelapse)) else (2, 2)
        )

        for t, frame_labels in frames:
            if t not in self._timed_spot_trees:
                continue

            tree, spot_centroids, spot_ids = self._timed_spot_trees[t]

            clouds = extract_all_point_clouds(
                frame_labels,
                num_points=2048,
                min_size=min_size,
            )

            for cloud in clouds:
                features = compute_single_shape_features(cloud, self._cal)

                if features.eigenvalues is None:
                    continue

                quality = math.pow(
                    float(features.eigenvalues[0])
                    * float(features.eigenvalues[1])
                    * float(features.eigenvalues[2]),
                    1.0 / 3.0,
                )

                centroid = features.centroid
                dist, index = tree.query(centroid)

                if dist < quality:
                    matched_spot_id = spot_ids[index]

                    props = {}
                    if features.eccentricity is not None:
                        props["eccentricity_comp_1"] = features.eccentricity[0]
                        props["eccentricity_comp_2"] = features.eccentricity[1]
                        props["eccentricity_comp_3"] = features.eccentricity[2]
                    if features.surface_area is not None:
                        props["surface_area"] = features.surface_area
                    if features.cell_axis_angles is not None:
                        props["cell_axis_z"] = features.cell_axis_angles[0]
                        props["cell_axis_y"] = features.cell_axis_angles[1]
                        props["cell_axis_x"] = features.cell_axis_angles[2]

                    radius = quality * math.pow(
                        self._cal[0] * self._cal[1] * self._cal[2],
                        1.0 / 3.0,
                    )
                    props["quality"] = quality
                    props["computed_radius"] = radius

                    self._spot_shape_features[matched_spot_id] = props

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
        z, y, x = (int(p / c) for p, c in zip(position, self._cal))

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

    def _match_shape_to_spot(self, spot_id: int) -> Optional[Dict]:
        """Get precomputed shape features for a spot matched via KDTree.

        During _compute_shape_features, each point cloud centroid from
        segmentation was matched to the nearest XML spot centroid using
        KDTree lookup (with quality threshold). This method simply
        retrieves those precomputed results.
        """
        return self._spot_shape_features.get(spot_id)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Assemble all features into a single DataFrame.

        Splits each track into tracklets along the lineage tree so that
        dividing tracks have separate tracklet IDs per sub-lineage.
        Non-dividing tracks get a single tracklet.

        Returns:
            DataFrame with columns:
                - Track_ID: unique tracklet ID (sub-lineage)
                - TrackMate_Track_ID: original TrackMate track ID
                - Generation_ID: 0 for mother, 1 for daughter, etc.
                - Tracklet_Number: sequential tracklet index within track
                - t, z, y, x: spot coordinates
                - Dividing, Number_Dividing: mitosis info
                - Shape features, dynamic features, track-level features
        """
        if self.is_master_xml:
            return self._master_dataframe()

        rows = []
        tracklet_counter = 0

        for track_id in self.xml.filtered_track_ids:
            track = self.xml.tracks.get(track_id)
            if track is None:
                continue

            is_dividing = self.xml.is_dividing(track_id)
            num_splits = track.num_splits
            tracklets = self._get_tracklets(track_id)

            for tlet in tracklets:
                tlet["tracklet_id"] = tracklet_counter
                tracklet_counter += 1

                spot_ids_sorted = tlet["spot_ids"]

                # MSD per tracklet (positions calibrated to physical units)
                positions = []
                for sid in spot_ids_sorted:
                    spot = self.xml.spots[sid]
                    positions.append([
                        spot.z * self._cal[0],
                        spot.y * self._cal[1],
                        spot.x * self._cal[2],
                    ])
                positions = np.array(positions)
                tracklet_msd = compute_msd(positions)

                prev_pos = None
                prev_prev_pos = None

                for i, sid in enumerate(spot_ids_sorted):
                    spot = self.xml.spots[sid]
                    pos = (spot.z, spot.y, spot.x)

                    dt = compute_dt(spot.frame, self._variable_t_calibration)

                    if prev_pos is not None:
                        # Speed in calibrated units per unit time
                        speed = compute_speed(pos, prev_pos, self._cal) / dt
                    else:
                        speed = 0.0

                    # Acceleration = ‖x[i] − 2 x[i-1] + x[i-2]‖ / dt²
                    # (second-difference of position, matching NapaTrackMater)
                    if prev_prev_pos is not None:
                        diff2 = (
                            np.array(pos) - 2.0 * np.array(prev_pos)
                            + np.array(prev_prev_pos)
                        ) * np.array(self._cal)
                        accel = float(np.linalg.norm(diff2)) / (dt * dt)
                    else:
                        accel = 0.0

                    if prev_pos is not None:
                        m_z, m_y, m_x = compute_motion_angles(pos, prev_pos)
                    else:
                        m_z, m_y, m_x = 0.0, 0.0, 0.0

                    r_z, r_y, r_x = compute_radial_angles(np.array(pos))

                    dist_mask, _ = self._get_boundary_distance(spot.frame, pos)
                    local_density = self._get_local_density(spot.frame, pos)

                    shape_props = self._match_shape_to_spot(sid)
                    if shape_props is not None:
                        ecc1 = shape_props.get("eccentricity_comp_1", np.nan)
                        ecc2 = shape_props.get("eccentricity_comp_2", np.nan)
                        ecc3 = shape_props.get("eccentricity_comp_3", np.nan)
                        surf_area = shape_props.get("surface_area", np.nan)
                        axis_z = shape_props.get("cell_axis_z", np.nan)
                        axis_y = shape_props.get("cell_axis_y", np.nan)
                        axis_x = shape_props.get("cell_axis_x", np.nan)
                    else:
                        ecc1 = ecc2 = ecc3 = np.nan
                        surf_area = np.nan
                        axis_z = axis_y = axis_x = np.nan

                    row = {
                        "Track_ID": tlet["tracklet_id"],
                        "TrackMate_Track_ID": track_id,
                        "Generation_ID": tlet["generation"],
                        "Tracklet_Number": tlet["tracklet_number"],
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
                        "MSD": tracklet_msd,
                        "Track_Displacement": track.displacement,
                        "Total_Track_Distance": track.total_distance,
                        "Max_Track_Distance": track.max_distance,
                        "Track_Duration": track.duration,
                        "Total_Intensity": spot.total_intensity,
                        "Mean_Intensity": spot.mean_intensity,
                    }
                    rows.append(row)

                    prev_prev_pos = prev_pos
                    prev_pos = pos

        return pd.DataFrame(rows)

    def _master_dataframe(self) -> pd.DataFrame:
        """Build the feature DataFrame directly from master-XML spot attrs.

        No recomputation: every dynamic / shape feature is read from the
        pre-populated ``SpotData.master`` dict. Tracklet lineage is still
        computed from the edge graph so ``Track_ID``, ``Generation_ID``,
        and ``Tracklet_Number`` remain consistent with the recompute path.
        """
        rows = []
        tracklet_counter = 0

        def g(m: Dict, key: str, default=np.nan):
            return m.get(key, default) if m is not None else default

        for track_id in self.xml.filtered_track_ids:
            track = self.xml.tracks.get(track_id)
            if track is None:
                continue

            is_dividing = self.xml.is_dividing(track_id)
            tracklets = self._get_tracklets(track_id)

            for tlet in tracklets:
                tlet["tracklet_id"] = tracklet_counter
                tracklet_counter += 1

                for sid in tlet["spot_ids"]:
                    spot = self.xml.spots[sid]
                    m = spot.master or {}

                    row = {
                        "Track_ID": tlet["tracklet_id"],
                        "TrackMate_Track_ID": track_id,
                        "Generation_ID": int(g(m, "generation_id", tlet["generation"])),
                        "Tracklet_Number": tlet["tracklet_number"],
                        "Unique_ID": int(g(m, "unique_id", sid)),
                        "t": spot.frame,
                        "z": spot.z,
                        "y": spot.y,
                        "x": spot.x,
                        "Dividing": int(g(m, "dividing", int(is_dividing))),
                        "Number_Dividing": int(g(m, "number_dividing", track.num_splits)),
                        "Radius": spot.radius,
                        "Eccentricity_Comp_First": g(m, "cloud_eccentricity_comp_first"),
                        "Eccentricity_Comp_Second": g(m, "cloud_eccentricity_comp_second"),
                        "Eccentricity_Comp_Third": g(m, "cloud_eccentricity_comp_third"),
                        "Local_Cell_Density": g(m, "local_density", 0),
                        "Surface_Area": g(m, "cloud_surfacearea"),
                        "Speed": g(m, "speed", 0.0),
                        "Motion_Angle_Z": g(m, "motion_angle_z", 0.0),
                        "Motion_Angle_Y": g(m, "motion_angle_y", 0.0),
                        "Motion_Angle_X": g(m, "motion_angle_x", 0.0),
                        "Acceleration": g(m, "acceleration", 0.0),
                        "Distance_Cell_mask": g(m, "distance_cell_mask", 0.0),
                        "Radial_Angle_Z": g(m, "radial_angle_z_key", 0.0),
                        "Radial_Angle_Y": g(m, "radial_angle_y_key", 0.0),
                        "Radial_Angle_X": g(m, "radial_angle_x_key", 0.0),
                        "Cell_Axis_Z": g(m, "cell_axis_z_key"),
                        "Cell_Axis_Y": g(m, "cell_axis_y_key"),
                        "Cell_Axis_X": g(m, "cell_axis_x_key"),
                        "MSD": g(m, "msd", 0.0),
                        "Track_Displacement": g(m, "track_displacement", track.displacement),
                        "Total_Track_Distance": g(
                            m, "total_distance_traveled", track.total_distance
                        ),
                        "Max_Track_Distance": g(
                            m, "max_distance_traveled", track.max_distance
                        ),
                        "Track_Duration": g(m, "track_duration", track.duration),
                        "Total_Intensity": spot.total_intensity,
                        "Mean_Intensity": spot.mean_intensity,
                    }
                    rows.append(row)

        return pd.DataFrame(rows)

    def _get_tracklets(self, track_id: int) -> List[Dict]:
        """
        Split a track into tracklets along its lineage tree.

        A tracklet is a linear segment between division events (or
        between root/division and leaf/division). Each tracklet gets
        a unique ID, a generation number, and a sequential tracklet
        number within the track.

        Returns:
            List of dicts, each with keys:
                - tracklet_id: unique sequential ID
                - generation: 0 for mother, 1 for daughters, etc.
                - tracklet_number: sequential within the track
                - spot_ids: ordered list of spot IDs in this tracklet
        """
        track = self.xml.tracks.get(track_id)
        if track is None:
            return []

        # If non-dividing, the whole track is one tracklet
        if track.num_splits == 0:
            spot_ids = self.xml.get_all_spot_ids_for_track(track_id)
            spot_ids_sorted = sorted(
                spot_ids, key=lambda sid: self.xml.spots[sid].frame
            )
            return [
                {
                    "tracklet_id": None,  # assigned later
                    "generation": 0,
                    "tracklet_number": 0,
                    "spot_ids": spot_ids_sorted,
                }
            ]

        # For dividing tracks, walk the lineage tree
        roots = self.xml.get_root_ids(track_id)
        tracklets = []
        tracklet_number = 0

        def walk(spot_id, generation):
            nonlocal tracklet_number
            current_tracklet = [spot_id]
            current = spot_id

            while True:
                children = self.xml.edge_target_lookup.get(current, [])
                if len(children) == 0:
                    # Leaf — end this tracklet
                    break
                elif len(children) == 1:
                    # Linear — continue this tracklet
                    current = children[0]
                    current_tracklet.append(current)
                else:
                    # Division — end this tracklet, recurse into children
                    for child in children:
                        walk(child, generation + 1)
                    break

            # Sort by time
            current_tracklet.sort(key=lambda sid: self.xml.spots[sid].frame)
            tracklets.append(
                {
                    "tracklet_id": None,
                    "generation": generation,
                    "tracklet_number": tracklet_number,
                    "spot_ids": current_tracklet,
                }
            )
            tracklet_number += 1

        for root in roots:
            walk(root, 0)

        return tracklets

    def get_shape_features_only(self) -> pd.DataFrame:
        """Get only shape features (no tracking required)."""
        if self.seg_image is None:
            raise ValueError("seg_image is required for shape features")

        return compute_shape_features_for_image(
            self.seg_image,
            num_points=2048,
            calibration=self._cal,
        )

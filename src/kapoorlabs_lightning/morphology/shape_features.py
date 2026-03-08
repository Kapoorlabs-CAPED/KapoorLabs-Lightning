"""
Shape feature computation from point clouds and segmentation images.

Computes eccentricity (via eigendecomposition of covariance matrix),
surface area (via convex hull), and cell axis orientation from point
cloud representations of segmented objects.

Can be used standalone without any tracking or XML data — just provide
a labeled segmentation image (and optionally a raw intensity image).
"""

import numpy as np
import pandas as pd
import concurrent.futures
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from skimage.measure import regionprops

from .point_clouds import extract_all_point_clouds, PointCloudResult


@dataclass
class ShapeFeatures:
    """Shape features computed from a single point cloud."""

    label: int
    centroid: np.ndarray
    eccentricity: Optional[np.ndarray] = None  # 3 components (sqrt eigenvalues)
    eigenvectors: Optional[np.ndarray] = None  # (3, 3) orientation matrix
    eigenvalues: Optional[np.ndarray] = None  # 3 raw eigenvalues
    surface_area: Optional[float] = None
    cell_axis_angles: Optional[np.ndarray] = None  # (3,) angles in radians


def compute_eccentricity(
    point_cloud: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute eccentricity from point cloud covariance matrix.

    Args:
        point_cloud: (N, 3) array of points.

    Returns:
        Tuple of (eccentricities, eigenvectors, eigenvalues) or None
        if eigenvalues are negative.
    """
    if point_cloud.shape[0] < 4:
        return None

    cov_mat = np.cov(point_cloud, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

    if np.any(eigenvalues < 0):
        return None

    eigenvectors = eigenvectors[:, ::-1]
    eccentricities = np.sqrt(eigenvalues)

    return eccentricities, eigenvectors, eigenvalues


def compute_surface_area(point_cloud: np.ndarray) -> Optional[float]:
    """
    Compute surface area via convex hull of the point cloud.

    Args:
        point_cloud: (N, 3) array of points.

    Returns:
        Surface area or None if convex hull fails.
    """
    try:
        hull = ConvexHull(point_cloud)
        return float(hull.area)
    except (QhullError, ValueError):
        return None


def _eigenvector_to_angles(eigenvectors: np.ndarray) -> np.ndarray:
    """
    Convert the principal eigenvector to spherical angles (z, y, x).

    Args:
        eigenvectors: (3, 3) eigenvector matrix.

    Returns:
        (3,) array of angles in radians.
    """
    principal = eigenvectors[:, 0]
    norm = np.linalg.norm(principal)
    if norm < 1e-10:
        return np.zeros(3)
    principal = principal / norm

    angle_z = np.arccos(np.clip(principal[0], -1, 1))
    angle_y = np.arctan2(principal[1], principal[2])
    angle_x = np.arctan2(
        np.sqrt(principal[0] ** 2 + principal[1] ** 2), principal[2]
    )

    return np.array([angle_z, angle_y, angle_x])


def compute_shape_features(
    cloud_result: PointCloudResult,
    calibration: Optional[Tuple[float, float, float]] = None,
) -> ShapeFeatures:
    """
    Compute all shape features for a single point cloud.

    Args:
        cloud_result: PointCloudResult from point cloud extraction.
        calibration: Optional (z_cal, y_cal, x_cal) to scale
            eccentricity values.

    Returns:
        ShapeFeatures dataclass with all computed features.
    """
    points = cloud_result.points
    features = ShapeFeatures(
        label=cloud_result.label,
        centroid=cloud_result.centroid,
    )

    ecc_result = compute_eccentricity(points)
    if ecc_result is not None:
        eccentricities, eigenvectors, eigenvalues = ecc_result
        if calibration is not None:
            cal = np.array(calibration)
            eccentricities = eccentricities * cal

        features.eccentricity = eccentricities
        features.eigenvectors = eigenvectors
        features.eigenvalues = eigenvalues
        features.cell_axis_angles = _eigenvector_to_angles(eigenvectors)

    features.surface_area = compute_surface_area(points)
    if features.surface_area is not None and calibration is not None:
        voxel_area = calibration[0] * calibration[1]
        features.surface_area *= voxel_area

    return features


def compute_shape_features_for_image(
    label_image: np.ndarray,
    raw_image: Optional[np.ndarray] = None,
    num_points: int = 2048,
    min_size: Optional[Tuple[int, ...]] = None,
    calibration: Optional[Tuple[float, float, float]] = None,
    num_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute shape features for all objects in a labeled segmentation image.

    This is the main standalone API — no tracking or XML required.
    Supports single frames (2D YX or 3D ZYX) or timelapse (TYX or TZYX).

    Args:
        label_image: Integer-labeled segmentation image.
            Supported shapes: (Y, X), (Z, Y, X), (T, Y, X), (T, Z, Y, X).
        raw_image: Optional intensity image (same shape as label_image)
            for intensity feature extraction.
        num_points: Number of surface points to sample per object.
        min_size: Minimum size in each spatial dimension.
        calibration: (z_cal, y_cal, x_cal) voxel calibration.
        num_workers: Threads for parallel processing.

    Returns:
        DataFrame with columns:
            t (if timelapse), label, centroid_z, centroid_y, centroid_x,
            eccentricity_comp_1, eccentricity_comp_2, eccentricity_comp_3,
            surface_area, cell_axis_z, cell_axis_y, cell_axis_x,
            [mean_intensity, total_intensity if raw_image provided]
    """
    ndim = label_image.ndim
    is_timelapse = ndim == 4 or (ndim == 3 and label_image.shape[0] < label_image.shape[-1])

    # For 2D/3D single frame, wrap in list
    if ndim <= 3 and not is_timelapse:
        frames = [(0, label_image)]
        raw_frames = [(0, raw_image)] if raw_image is not None else None
    else:
        frames = [(t, label_image[t]) for t in range(label_image.shape[0])]
        raw_frames = (
            [(t, raw_image[t]) for t in range(raw_image.shape[0])]
            if raw_image is not None
            else None
        )

    if min_size is None:
        if ndim >= 3 and not (ndim == 3 and is_timelapse):
            min_size = (2, 2, 2)
        else:
            min_size = (2, 2)

    all_rows = []

    for t, frame_labels in frames:
        # Extract point clouds
        clouds = extract_all_point_clouds(
            frame_labels,
            num_points=num_points,
            min_size=min_size,
            num_workers=num_workers,
        )

        # Get intensity info if raw image available
        intensity_map = {}
        if raw_frames is not None:
            _, raw_frame = raw_frames[t]
            props = regionprops(frame_labels, intensity_image=raw_frame)
            for prop in props:
                intensity_map[prop.label] = {
                    "mean_intensity": float(prop.intensity_mean),
                    "total_intensity": float(
                        prop.intensity_mean * prop.area
                    ),
                }

        # Compute shape features
        for cloud in clouds:
            features = compute_shape_features(cloud, calibration)

            row = {"t": t, "label": features.label}

            # Centroid
            if len(features.centroid) == 3:
                row["centroid_z"] = features.centroid[0]
                row["centroid_y"] = features.centroid[1]
                row["centroid_x"] = features.centroid[2]
            elif len(features.centroid) == 2:
                row["centroid_z"] = 0.0
                row["centroid_y"] = features.centroid[0]
                row["centroid_x"] = features.centroid[1]

            # Eccentricity
            if features.eccentricity is not None:
                row["eccentricity_comp_1"] = features.eccentricity[0]
                row["eccentricity_comp_2"] = features.eccentricity[1]
                row["eccentricity_comp_3"] = features.eccentricity[2]
            else:
                row["eccentricity_comp_1"] = np.nan
                row["eccentricity_comp_2"] = np.nan
                row["eccentricity_comp_3"] = np.nan

            # Surface area
            row["surface_area"] = (
                features.surface_area
                if features.surface_area is not None
                else np.nan
            )

            # Cell axis angles
            if features.cell_axis_angles is not None:
                row["cell_axis_z"] = features.cell_axis_angles[0]
                row["cell_axis_y"] = features.cell_axis_angles[1]
                row["cell_axis_x"] = features.cell_axis_angles[2]
            else:
                row["cell_axis_z"] = np.nan
                row["cell_axis_y"] = np.nan
                row["cell_axis_x"] = np.nan

            # Intensity
            if features.label in intensity_map:
                row["mean_intensity"] = intensity_map[features.label][
                    "mean_intensity"
                ]
                row["total_intensity"] = intensity_map[features.label][
                    "total_intensity"
                ]

            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # Drop t column if single frame
    if not is_timelapse and "t" in df.columns:
        df = df.drop(columns=["t"])

    return df

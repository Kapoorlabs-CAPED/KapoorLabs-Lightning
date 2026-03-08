"""
Point cloud extraction from segmentation images.

Converts labeled segmentation regions into point cloud representations
using marching cubes (3D) or contour extraction (2D).
"""

import numpy as np
import concurrent.futures
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

from skimage.measure import regionprops, marching_cubes, find_contours

try:
    import trimesh
except ImportError:
    trimesh = None


@dataclass
class PointCloudResult:
    """Result of point cloud extraction for a single labeled region."""

    label: int
    centroid: np.ndarray
    points: np.ndarray  # (N, 3) array of sampled surface points


def extract_point_cloud(
    binary_image: np.ndarray,
    num_points: int,
    label: int,
    centroid: np.ndarray,
    min_size: Optional[Tuple[int, ...]] = None,
) -> Optional[PointCloudResult]:
    """
    Extract a point cloud from a single binary region.

    For 3D images, uses marching cubes to generate a mesh surface and
    samples points uniformly from it. For 2D images, extracts contours
    and samples points along them.

    Args:
        binary_image: 2D or 3D binary image of a single region.
        num_points: Number of surface points to sample.
        label: Integer label of this region.
        centroid: Centroid coordinates of this region.
        min_size: Minimum required size in each dimension. Regions
            smaller than this are skipped.

    Returns:
        PointCloudResult or None if the region is too small or
        extraction fails.
    """
    if min_size is not None:
        if any(
            binary_image.shape[j] < min_size[j]
            for j in range(len(binary_image.shape))
        ):
            return None

    if binary_image.ndim == 3:
        if trimesh is None:
            raise ImportError(
                "trimesh is required for 3D point cloud extraction. "
                "Install it with: pip install trimesh"
            )
        try:
            vertices, faces, _, _ = marching_cubes(binary_image)
        except (RuntimeError, ValueError):
            return None

        mesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
        points = np.asarray(mesh_obj.sample(num_points).data)

    elif binary_image.ndim == 2:
        contours = find_contours(binary_image, level=0.5)
        if not contours:
            return None

        points = np.vstack(contours)
        if points.shape[0] > num_points:
            indices = np.random.choice(
                points.shape[0], num_points, replace=False
            )
            points = points[indices]

        # Add z=0 dimension for consistency: (N, 2) → (N, 3)
        points = np.hstack(
            (np.zeros((points.shape[0], 1)), points)
        )
    else:
        return None

    return PointCloudResult(
        label=label,
        centroid=centroid,
        points=points,
    )


def _process_region(prop, num_points, min_size):
    """Process a single regionprop into a point cloud."""
    binary_image = prop.image
    label = prop.label
    centroid = np.asarray(prop.centroid)
    return extract_point_cloud(
        binary_image, num_points, label, centroid, min_size
    )


def extract_all_point_clouds(
    label_image: np.ndarray,
    num_points: int = 2048,
    min_size: Optional[Tuple[int, ...]] = None,
    num_workers: Optional[int] = None,
) -> List[PointCloudResult]:
    """
    Extract point clouds for all labeled regions in a segmentation image.

    Processes a single 2D or 3D labeled image (not a timelapse — for
    timelapse data, call this per-frame).

    Args:
        label_image: 2D (YX) or 3D (ZYX) integer-labeled segmentation.
        num_points: Number of surface points to sample per region.
        min_size: Minimum size in each dimension to keep a region.
        num_workers: Number of threads for parallel extraction.
            Defaults to os.cpu_count().

    Returns:
        List of PointCloudResult objects, one per valid region.
    """
    if num_workers is None:
        num_workers = os.cpu_count()

    properties = regionprops(label_image)
    results = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:
        futures = [
            executor.submit(_process_region, prop, num_points, min_size)
            for prop in properties
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    return results

"""
Dynamic feature computation for cell tracks.

Computes speed, acceleration, motion angles, radial angles, MSD,
and other dynamic features from tracked cell positions over time.
"""

import math
import numpy as np
from typing import Dict, Optional, Tuple

# Feature name constants
SHAPE_FEATURES = [
    "Radius",
    "Eccentricity_Comp_First",
    "Eccentricity_Comp_Second",
    "Eccentricity_Comp_Third",
    "Local_Cell_Density",
    "Surface_Area",
]

DYNAMIC_FEATURES = [
    "Speed",
    "Motion_Angle_Z",
    "Motion_Angle_Y",
    "Motion_Angle_X",
    "Acceleration",
    "Distance_Cell_mask",
    "Radial_Angle_Z",
    "Radial_Angle_Y",
    "Radial_Angle_X",
    "Cell_Axis_Z",
    "Cell_Axis_Y",
    "Cell_Axis_X",
]

SHAPE_DYNAMIC_FEATURES = SHAPE_FEATURES + DYNAMIC_FEATURES

IDENTITY_FEATURES = [
    "Track_ID",
    "t",
    "z",
    "y",
    "x",
    "Dividing",
    "Number_Dividing",
]

TRACK_TYPE_FEATURES = ["MSD"]

ALL_FEATURES = IDENTITY_FEATURES + SHAPE_DYNAMIC_FEATURES + TRACK_TYPE_FEATURES


def compute_speed(
    pos_current: np.ndarray,
    pos_previous: np.ndarray,
    calibration: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    Compute speed between two positions.

    Args:
        pos_current: Current position (z, y, x).
        pos_previous: Previous position (z, y, x).
        calibration: (z_cal, y_cal, x_cal) voxel calibration.

    Returns:
        Euclidean distance between positions (speed).
    """
    diff = np.array(pos_current) - np.array(pos_previous)
    cal = np.array(calibration)
    calibrated_diff = diff * cal
    return float(np.linalg.norm(calibrated_diff))


def compute_dt(
    frame: int,
    variable_t_calibration: Optional[Dict[int, float]] = None,
    default: float = 1.0,
) -> float:
    """Look up the time-per-frame interval for a given frame.

    The mapping uses frame-index upper bounds as keys, e.g.
    ``{160: 182, 260: 283}`` means: for frame < 160 use dt=182,
    for 160 <= frame < 260 use dt=283. For frames at or beyond the
    largest key, the last value is carried over.

    Args:
        frame: Frame index.
        variable_t_calibration: Mapping of upper-bound frame -> dt.
            If None or empty, returns ``default``.
        default: Fallback dt when no mapping is provided.

    Returns:
        The dt value for the given frame.
    """
    if not variable_t_calibration:
        return default
    sorted_items = sorted(
        ((int(k), float(v)) for k, v in variable_t_calibration.items()),
        key=lambda kv: kv[0],
    )
    for upper, dt in sorted_items:
        if frame < upper:
            return dt
    return sorted_items[-1][1]


def compute_acceleration(
    speed_current: float,
    speed_previous: float,
    dt: float = 1.0,
) -> float:
    """
    Compute acceleration from consecutive speed values.

    Args:
        speed_current: Speed at current timepoint.
        speed_previous: Speed at previous timepoint.
        dt: Time interval between measurements.

    Returns:
        Acceleration value.
    """
    if dt == 0:
        return 0.0
    return (speed_current - speed_previous) / dt


def compute_motion_angles(
    pos_current: np.ndarray,
    pos_previous: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute motion vector angles relative to z, y, x axes.

    Args:
        pos_current: Current position (z, y, x).
        pos_previous: Previous position (z, y, x).

    Returns:
        Tuple of (angle_z, angle_y, angle_x) in degrees.
    """
    motion_vec = np.array(pos_current) - np.array(pos_previous)
    norm = np.linalg.norm(motion_vec)
    if norm < 1e-10:
        return (0.0, 0.0, 0.0)

    return (
        angular_change_z(motion_vec),
        angular_change_y(motion_vec),
        angular_change_x(motion_vec),
    )


def compute_radial_angles(
    position: np.ndarray,
    reference: np.ndarray = None,
) -> Tuple[float, float, float]:
    """
    Compute radial angles from a reference point (default: origin).

    Args:
        position: Current position (z, y, x).
        reference: Reference point. Defaults to origin.

    Returns:
        Tuple of (angle_z, angle_y, angle_x) in degrees.
    """
    if reference is None:
        reference = np.zeros_like(position)
    radial_vec = np.array(position) - np.array(reference)
    norm = np.linalg.norm(radial_vec)
    if norm < 1e-10:
        return (0.0, 0.0, 0.0)

    return (
        angular_change_z(radial_vec),
        angular_change_y(radial_vec),
        angular_change_x(radial_vec),
    )


def compute_msd(
    positions: np.ndarray,
) -> float:
    """
    Compute mean squared displacement from a sequence of positions.

    Args:
        positions: (N, 3) array of (z, y, x) positions over time.

    Returns:
        Mean squared displacement from the starting position.
    """
    if len(positions) < 2:
        return 0.0
    origin = positions[0]
    displacements = positions - origin
    squared_disps = np.sum(displacements**2, axis=1)
    return float(np.mean(squared_disps))


def angular_change_z(vec_cell: np.ndarray) -> float:
    """Angle between vector and z-axis (last component) in degrees."""
    vec = np.asarray(vec_cell, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return 0.0
    vec = vec / norm
    unit = np.zeros(len(vec))
    unit[-1] = 1.0
    theta = np.arccos(np.clip(np.dot(vec, unit), -1.0, 1.0))
    return float(np.rad2deg(theta))


def angular_change_y(vec_cell: np.ndarray) -> float:
    """Angle between vector and y-axis (second-to-last component) in degrees."""
    vec = np.asarray(vec_cell, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return 0.0
    vec = vec / norm
    unit = np.zeros(len(vec))
    unit[-2] = 1.0
    theta = np.arccos(np.clip(np.dot(vec, unit), -1.0, 1.0))
    return float(np.rad2deg(theta))


def angular_change_x(vec_cell: np.ndarray) -> float:
    """Angle between vector and x-axis (first component) in degrees."""
    vec = np.asarray(vec_cell, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return 0.0
    vec = vec / norm
    unit = np.zeros(len(vec))
    unit[0] = 1.0
    theta = np.arccos(np.clip(np.dot(vec, unit), -1.0, 1.0))
    return float(np.rad2deg(theta))

"""Master-corrected-graph builder: enrich a track DiGraph with the per-spot
shape / dynamics / intensity features TrackMate's master XML carries, so the
graph itself becomes the single source of truth for downstream feature
DataFrames.

Mirrors what the TrackMate path produces via ``TrackVectors._master_dataframe``
— except features land as ``networkx`` node attributes instead of XML
``<SpotFeature>`` tags, and ``write_master_graph`` / ``read_master_graph``
play the role of ``write_trackmate_xml`` (the ``master_<original>.xml``
on-disk artefact). The bridge's
:func:`kapoorlabs_lightning.tracking.trackastra_bridge.graph_to_dataframe`
reads these attributes when present (no recomputation) and falls back
to on-the-fly computation otherwise. Run order:

    Trackastra → DiGraph
        │
        ▼ oneat_correct_graph(G, events_csv)         ← divisions repaired
        ▼ enrich_graph_with_shape_features(G, seg)   ← shape + intensity cached
        ▼ enrich_graph_with_dynamics(G, calibration) ← speed/MSD/etc. cached
        ▼ write_master_graph(G, "master.json")       ← persisted to disk
        ▼ graph_to_dataframe(G)                      ← consumer table
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import networkx as nx
import numpy as np
from skimage.measure import regionprops

from .track_features import (
    compute_dt,
    compute_motion_angles,
    compute_msd,
    compute_radial_angles,
    compute_speed,
)


# These match the column names TrackVectors writes — keep one place
# updated and ``graph_to_dataframe`` picks them up automatically.
_SHAPE_NODE_ATTRS = (
    "Radius",
    "Eccentricity_Comp_First",
    "Eccentricity_Comp_Second",
    "Eccentricity_Comp_Third",
    "Surface_Area",
    "Cell_Axis_Z",
    "Cell_Axis_Y",
    "Cell_Axis_X",
    "Total_Intensity",
    "Mean_Intensity",
)


def _shape_props(
    seg_frame: np.ndarray,
    raw_frame: Optional[np.ndarray],
    label: int,
    *,
    calibration: tuple[float, float, float],
) -> dict:
    """Compute the master shape / intensity columns for one (t, label) cell.

    Empty dict if the label is absent from ``seg_frame``.
    """
    props = regionprops(
        (seg_frame == label).astype(np.uint16),
        intensity_image=raw_frame,
    )
    if not props:
        return {}
    r = props[0]
    voxel_volume = float(np.prod(calibration))
    volume = float(r.area) * voxel_volume
    radius = float(np.cbrt(volume * 3.0 / (4.0 * np.pi)))

    try:
        eigs = sorted(
            np.asarray(r.inertia_tensor_eigvals, dtype=float).tolist(),
            reverse=True,
        )
        while len(eigs) < 3:
            eigs.append(np.nan)
    except (AttributeError, ValueError):
        eigs = [np.nan, np.nan, np.nan]

    axes = [
        float(np.sqrt(max(e, 0.0)) * 4.0) if not np.isnan(e) else np.nan
        for e in eigs
    ]

    out = {
        "Radius": radius,
        "Eccentricity_Comp_First": eigs[0],
        "Eccentricity_Comp_Second": eigs[1],
        "Eccentricity_Comp_Third": eigs[2],
        "Cell_Axis_Z": axes[0],
        "Cell_Axis_Y": axes[1],
        "Cell_Axis_X": axes[2],
        # Cheap surface-area estimate: perimeter voxels × dy·dx
        # (matching what TrackVectors writes for 3D ROIs).
        "Surface_Area": float(r.area) * float(calibration[-1] * calibration[-2]),
    }
    if raw_frame is not None:
        out["Mean_Intensity"] = float(r.mean_intensity)
        out["Total_Intensity"] = float(r.area * r.mean_intensity)
    return out


def enrich_graph_with_shape_features(
    graph: nx.DiGraph,
    *,
    seg_image: np.ndarray,
    raw_image: Optional[np.ndarray] = None,
    calibration: tuple[float, float, float] = (1.0, 1.0, 1.0),
    frame_attribute: str = "time",
    label_attribute: str = "label",
    overwrite: bool = False,
) -> nx.DiGraph:
    """Cache per-spot shape / intensity features onto graph nodes.

    Parameters
    ----------
    graph
        Trackastra-shaped DiGraph; each node must have
        ``frame_attribute`` and ``label_attribute`` set (Trackastra
        does this by default; the ``oneat_correct_graph`` output
        preserves them).
    seg_image
        ``(T, Z, Y, X)`` or ``(T, Y, X)`` segmentation stack — the
        per-frame label image whose IDs match the node ``label``.
    raw_image
        Optional matching-shape raw stack for intensity columns.
    calibration
        ``(dz, dy, dx)`` voxel size; truncated for 2D.
    overwrite
        If False (default), nodes that already have a ``Radius``
        attribute are skipped — useful for re-running enrichment
        after only some divisions were corrected.

    Returns
    -------
    nx.DiGraph
        Copy of ``graph`` with every node carrying the
        :data:`_SHAPE_NODE_ATTRS` keys (or NaN where the seg label
        couldn't be found).
    """
    G = graph.copy()

    # Bucket nodes by frame so we only load each seg/raw slice once.
    by_frame: dict[int, list] = {}
    for n, data in G.nodes(data=True):
        t = int(
            data.get(frame_attribute, data.get("t", data.get("time")))
        )
        by_frame.setdefault(t, []).append(n)

    for t, nodes in by_frame.items():
        if t < 0 or t >= seg_image.shape[0]:
            continue
        seg_frame = seg_image[t]
        raw_frame = (
            raw_image[t]
            if raw_image is not None and 0 <= t < raw_image.shape[0]
            else None
        )

        for n in nodes:
            data = G.nodes[n]
            if not overwrite and "Radius" in data:
                continue
            if label_attribute not in data:
                continue
            label = int(data[label_attribute])
            props = _shape_props(
                seg_frame, raw_frame, label, calibration=calibration,
            )
            for k in _SHAPE_NODE_ATTRS:
                if k in props:
                    data[k] = props[k]
                elif k not in data:
                    data[k] = float("nan")
    return G


# ============================================================ dynamics cache

_DYNAMICS_NODE_ATTRS = (
    "Speed",
    "Acceleration",
    "Motion_Angle_Z",
    "Motion_Angle_Y",
    "Motion_Angle_X",
    "Radial_Angle_Z",
    "Radial_Angle_Y",
    "Radial_Angle_X",
    "MSD",
    "Track_Displacement",
    "Total_Track_Distance",
    "Max_Track_Distance",
    "Track_Duration",
)


def enrich_graph_with_dynamics(
    graph: nx.DiGraph,
    *,
    calibration: tuple[float, float, float] = (1.0, 1.0, 1.0),
    variable_t_calibration: Optional[dict] = None,
    frame_attribute: str = "time",
    coords_attribute: str = "coords",
    overwrite: bool = False,
) -> nx.DiGraph:
    """Cache per-spot dynamics (Speed/Accel/Angles) + tracklet MSD + track
    aggregates onto graph nodes — same set ``TrackVectors._master_dataframe``
    reads from ``spot.master``.

    Must be called *after* divisions are correct (so MSD is computed
    per current tracklet, not per pre-correction tracklet).
    """
    from .trackastra_bridge import walk_tracklets

    G = graph.copy()
    cal = np.asarray(calibration, dtype=np.float64)

    # Per-component aggregates (Track_Displacement etc.).
    comp_aggs: dict[int, dict] = {}
    for comp_id, comp in enumerate(nx.weakly_connected_components(G)):
        nodes_sorted = sorted(
            comp,
            key=lambda nn: int(
                G.nodes[nn].get(frame_attribute, G.nodes[nn].get("t",
                G.nodes[nn].get("time")))
            ),
        )
        if not nodes_sorted:
            comp_aggs[comp_id] = dict(
                displacement=0.0, total_distance=0.0,
                max_distance=0.0, duration=0,
            )
            continue
        first, last = nodes_sorted[0], nodes_sorted[-1]
        p_first = np.asarray(G.nodes[first][coords_attribute]) * cal[
            : len(G.nodes[first][coords_attribute])
        ]
        p_last = np.asarray(G.nodes[last][coords_attribute]) * cal[
            : len(G.nodes[last][coords_attribute])
        ]
        total = 0.0; max_step = 0.0
        for u, v in G.edges():
            if u not in comp:
                continue
            pu = np.asarray(G.nodes[u][coords_attribute]) * cal[
                : len(G.nodes[u][coords_attribute])
            ]
            pv = np.asarray(G.nodes[v][coords_attribute]) * cal[
                : len(G.nodes[v][coords_attribute])
            ]
            step = float(np.linalg.norm(pv - pu))
            total += step
            if step > max_step:
                max_step = step
        comp_aggs[comp_id] = dict(
            displacement=float(np.linalg.norm(p_last - p_first)),
            total_distance=total,
            max_distance=max_step,
            duration=int(G.nodes[last].get(frame_attribute, G.nodes[last].get("t", G.nodes[last].get("time"))) -
                         G.nodes[first].get(frame_attribute, G.nodes[first].get("t", G.nodes[first].get("time")))),
        )

    # Per-tracklet dynamics.
    for tl in walk_tracklets(G, frame_attribute=frame_attribute):
        agg = comp_aggs[tl.component_id]
        positions_phys = []
        for n in tl.nodes:
            coords = np.asarray(G.nodes[n][coords_attribute], dtype=np.float64)
            positions_phys.append(coords * cal[: coords.shape[0]])
        positions_phys = np.asarray(positions_phys)
        msd_val = compute_msd(positions_phys) if len(positions_phys) > 1 else 0.0

        prev_pos = None
        prev_prev_pos = None
        for n in tl.nodes:
            data = G.nodes[n]
            if not overwrite and "Speed" in data:
                # already cached — still set track-aggregate fields for
                # downstream consistency.
                data.setdefault("MSD", msd_val)
                data.setdefault("Track_Displacement", agg["displacement"])
                data.setdefault("Total_Track_Distance", agg["total_distance"])
                data.setdefault("Max_Track_Distance", agg["max_distance"])
                data.setdefault("Track_Duration", agg["duration"])
                prev_prev_pos = prev_pos
                prev_pos = np.asarray(data[coords_attribute]) * cal[
                    : len(data[coords_attribute])
                ]
                continue

            coords = np.asarray(data[coords_attribute], dtype=np.float64)
            pos_phys = coords * cal[: coords.shape[0]]
            if pos_phys.shape[0] == 2:
                pos_phys = np.asarray([0.0, pos_phys[0], pos_phys[1]])

            t_int = int(data.get(frame_attribute, data.get("t", data.get("time"))))
            dt = compute_dt(t_int, variable_t_calibration)

            if prev_pos is not None:
                speed = compute_speed(tuple(pos_phys), tuple(prev_pos),
                                      (1.0, 1.0, 1.0)) / dt
            else:
                speed = 0.0
            if prev_prev_pos is not None:
                d2 = pos_phys - 2.0 * prev_pos + prev_prev_pos
                accel = float(np.linalg.norm(d2)) / (dt * dt)
            else:
                accel = 0.0
            if prev_pos is not None:
                m_z, m_y, m_x = compute_motion_angles(
                    tuple(pos_phys), tuple(prev_pos),
                )
            else:
                m_z, m_y, m_x = 0.0, 0.0, 0.0
            r_z, r_y, r_x = compute_radial_angles(pos_phys)

            data["Speed"] = float(speed)
            data["Acceleration"] = float(accel)
            data["Motion_Angle_Z"] = float(m_z)
            data["Motion_Angle_Y"] = float(m_y)
            data["Motion_Angle_X"] = float(m_x)
            data["Radial_Angle_Z"] = float(r_z)
            data["Radial_Angle_Y"] = float(r_y)
            data["Radial_Angle_X"] = float(r_x)
            data["MSD"] = float(msd_val)
            data["Track_Displacement"] = float(agg["displacement"])
            data["Total_Track_Distance"] = float(agg["total_distance"])
            data["Max_Track_Distance"] = float(agg["max_distance"])
            data["Track_Duration"] = int(agg["duration"])

            prev_prev_pos = prev_pos
            prev_pos = pos_phys
    return G


# ============================================================ on-disk master

def _node_to_jsonable(data: dict) -> dict:
    """Convert numpy-typed node attrs into JSON-serialisable scalars."""
    out = {}
    for k, v in data.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v) if isinstance(v, np.floating) else int(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def write_master_graph(
    graph: nx.DiGraph,
    path: Union[str, Path],
) -> Path:
    """Persist a master-enriched graph to disk in networkx node-link JSON.

    The on-disk artefact plays the same role as
    ``master_<original>.xml`` on the TrackMate side: everything Oneat /
    cellfate / curvature needs is included, no segmentation / raw images
    required to re-derive features at read time.
    """
    path = Path(path)
    # Build with payload data first so the explicit `id` / `source` /
    # `target` keys always win, even if a node/edge attr happens to
    # share one of those names (networkx node-link uses them as the
    # link identifiers).
    nodes = [
        {**_node_to_jsonable(data), "id": str(n)}
        for n, data in graph.nodes(data=True)
    ]
    edges = [
        {**_node_to_jsonable(data), "source": str(u), "target": str(v)}
        for u, v, data in graph.edges(data=True)
    ]
    payload = {
        "directed": True,
        "multigraph": False,
        "graph": dict(graph.graph),
        "nodes": nodes,
        "links": edges,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    return path


def read_master_graph(path: Union[str, Path]) -> nx.DiGraph:
    """Inverse of :func:`write_master_graph` — read a master graph JSON back
    into a ``nx.DiGraph`` with all cached spot attrs intact."""
    with Path(path).open() as fh:
        payload = json.load(fh)
    G = nx.DiGraph()
    G.graph.update(payload.get("graph", {}))
    for n in payload.get("nodes", []):
        node_id = n.pop("id")
        # Re-normalise coords back to a tuple (was a list in JSON).
        if "coords" in n and isinstance(n["coords"], list):
            n["coords"] = tuple(n["coords"])
        G.add_node(node_id, **n)
    for e in payload.get("links", []):
        u = e.pop("source"); v = e.pop("target")
        G.add_edge(u, v, **e)
    return G

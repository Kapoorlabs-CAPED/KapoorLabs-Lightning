"""Trackastra → KapoorLabs DataFrame bridge.

Trackastra emits a single ``networkx.DiGraph`` (a transformer-scored
candidate graph then solved by a greedy / ILP linker). KapoorLabs
downstream code (Oneat, cellfate, the curvature scripts) consumes the
``TrackVectors.to_dataframe`` schema:

    Track_ID, TrackMate_Track_ID, Generation_ID, Tracklet_Number,
    t, z, y, x, Dividing, Number_Dividing, plus per-spot shape /
    dynamics / intensity / track-level columns.

This bridge produces *exactly* that schema from a Trackastra graph so
the existing TrackMate-driven and Trackastra-driven flows share one
downstream stack:

    Trackastra ───┐
                  ├─→ DataFrame ─→ Oneat / Cellfate / Curvature
    TrackMate ────┘

The ``TrackMate_Track_ID`` column is preserved as a first-class field
even in the Trackastra path — it just maps to the connected-component
id of the graph (every lineage tree → one component → one
``TrackMate_Track_ID``).

The reverse, :func:`dataframe_to_graph`, lets Oneat-corrected
DataFrames be turned back into a Trackastra-compatible
``networkx.DiGraph`` so the same graph-level tools (Trackastra napari
viewer, ``apply_solution_graph_to_masks``) keep working.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd

from .track_features import (
    compute_motion_angles,
    compute_msd,
    compute_radial_angles,
    compute_speed,
    compute_dt,
)


# ============================================================ tracklet walk

@dataclass
class _Tracklet:
    component_id: int                            # → TrackMate_Track_ID
    tracklet_number: int                         # sequential within component
    generation: int                              # depth from component root
    nodes: list                                   # ordered list of graph nodes
    parent_tracklet_number: Optional[int] = None  # for round-trip + Oneat edits


def _component_root(G: nx.DiGraph, comp_nodes: set) -> object:
    """Pick the in-degree-0 node of a connected component.

    Trackastra's solver guarantees forward-in-time edges and at most
    one parent per node, so a connected component is a tree with one
    or more roots (sources, in_degree=0); we pick the earliest by
    frame. If there are no in-degree-0 nodes (cycle — shouldn't
    happen but be safe), fall back to the earliest-frame node.
    """
    roots = [n for n in comp_nodes if G.in_degree(n) == 0]
    if not roots:
        roots = list(comp_nodes)
    return min(roots, key=lambda n: G.nodes[n]["__t"])


def walk_tracklets(
    G: nx.DiGraph,
    *,
    frame_attribute: str = "time",
) -> list[_Tracklet]:
    """Split every connected component of ``G`` into linear tracklets.

    A tracklet starts at a root / after a division / after a
    gap-closing edge (Δt > 1) and ends before a division / at a
    leaf / before a gap-closing edge — the same rule Trackastra uses
    in :func:`trackastra.tracking.utils.ctc_tracklets`.

    Each tracklet gets a ``component_id`` (the connected-component
    index, used downstream as ``TrackMate_Track_ID``), a sequential
    ``tracklet_number`` within that component, and a ``generation``
    counting how many division edges lie between the tracklet's root
    and the component's root.
    """
    # Make sure every node carries a normalised ``__t`` we can sort by.
    for _, data in G.nodes(data=True):
        if "__t" not in data:
            if frame_attribute in data:
                data["__t"] = int(data[frame_attribute])
            elif "t" in data:
                data["__t"] = int(data["t"])
            elif "time" in data:
                data["__t"] = int(data["time"])
            else:
                raise KeyError(
                    f"Graph node {data} has no frame attribute "
                    f"({frame_attribute!r}, 't', or 'time')."
                )

    out: list[_Tracklet] = []
    for comp_id, comp in enumerate(nx.weakly_connected_components(G)):
        comp = set(comp)
        root = _component_root(G, comp)
        # BFS-walk from the root; each entry is (start_node, generation,
        # parent_tracklet_number).
        queue: deque = deque([(root, 0, None)])
        tracklet_in_comp = 0
        while queue:
            start, gen, parent_tnum = queue.popleft()
            nodes = [start]
            cur = start
            while True:
                succ = list(G.successors(cur))
                if len(succ) == 0:
                    break                                     # leaf
                if len(succ) >= 2:                            # division
                    for child in succ:
                        queue.append((child, gen + 1, tracklet_in_comp))
                    break
                # len(succ) == 1 — linear, possibly with a gap
                nxt = succ[0]
                if G.nodes[nxt]["__t"] - G.nodes[cur]["__t"] > 1:
                    # Gap-closing → start a fresh tracklet, same generation.
                    queue.append((nxt, gen, tracklet_in_comp))
                    break
                nodes.append(nxt)
                cur = nxt

            out.append(_Tracklet(
                component_id=comp_id,
                tracklet_number=tracklet_in_comp,
                generation=gen,
                nodes=nodes,
                parent_tracklet_number=parent_tnum,
            ))
            tracklet_in_comp += 1
    return out


# ============================================================ optional shape

def _label_to_slice(seg_frame: np.ndarray) -> dict[int, tuple]:
    """``{label: bbox slice}`` for one segmentation frame.

    Used so we can compute per-label shape props without re-scanning
    the whole frame for every cell.
    """
    from skimage.measure import regionprops
    return {int(r.label): r.slice for r in regionprops(seg_frame)}


def _shape_props_for(
    seg_frame: np.ndarray,
    label: int,
    *,
    bbox_cache: dict,
    calibration: tuple[float, float, float],
    raw_frame: Optional[np.ndarray] = None,
) -> dict:
    """Compute the shape / intensity columns for one (t, label) cell.

    Mirrors what ``TrackVectors._compute_shape_features`` writes per
    spot: ``Radius`` (equivalent sphere radius from volume),
    ``Eccentricity_Comp_*`` (sorted inertia eigenvalues), ``Surface_Area``
    (3D — count of perimeter voxels × surface element area; cheap
    approximation), ``Cell_Axis_*`` (principal-axis lengths), plus
    intensity stats when ``raw_frame`` is given.
    """
    from skimage.measure import regionprops

    if label not in bbox_cache:
        return {}
    sl = bbox_cache[label]
    sub_seg = seg_frame[sl] == label
    sub_raw = raw_frame[sl] if raw_frame is not None else None
    props_iter = regionprops(
        sub_seg.astype(np.uint16),
        intensity_image=sub_raw,
    )
    if not props_iter:
        return {}
    r = props_iter[0]

    voxel_volume = float(np.prod(calibration))
    volume = float(r.area) * voxel_volume
    radius = float(np.cbrt(volume * 3.0 / (4.0 * np.pi)))

    # inertia-tensor eigenvalues come pre-sorted descending in skimage
    try:
        eigs = sorted(
            np.asarray(r.inertia_tensor_eigvals, dtype=float).tolist(),
            reverse=True,
        )
        # pad to length 3 for 2D ROIs
        while len(eigs) < 3:
            eigs.append(np.nan)
    except (AttributeError, ValueError):
        eigs = [np.nan, np.nan, np.nan]

    # Cheap principal-axis lengths from the inertia eigenvalues.
    axes = [
        float(np.sqrt(max(e, 0.0)) * 4.0) if not np.isnan(e) else np.nan
        for e in eigs
    ]

    out: dict = {
        "Radius": radius,
        "Eccentricity_Comp_First": eigs[0],
        "Eccentricity_Comp_Second": eigs[1],
        "Eccentricity_Comp_Third": eigs[2],
        "Cell_Axis_Z": axes[0],
        "Cell_Axis_Y": axes[1],
        "Cell_Axis_X": axes[2],
    }
    if raw_frame is not None:
        out["Mean_Intensity"] = float(r.mean_intensity)
        out["Total_Intensity"] = float(r.area * r.mean_intensity)
    return out


# ============================================================ public API

# Match the column order of TrackVectors.to_dataframe so downstream
# consumers (cellfate, oneat) don't care which path produced the table.
_COLUMNS = [
    "Track_ID", "TrackMate_Track_ID", "Generation_ID", "Tracklet_Number",
    "Parent_Tracklet_Number",                # extra — for the round-trip
    "t", "z", "y", "x",
    "Dividing", "Number_Dividing",
    "Radius",
    "Eccentricity_Comp_First", "Eccentricity_Comp_Second",
    "Eccentricity_Comp_Third",
    "Local_Cell_Density", "Surface_Area",
    "Speed",
    "Motion_Angle_Z", "Motion_Angle_Y", "Motion_Angle_X",
    "Acceleration",
    "Distance_Cell_mask",
    "Radial_Angle_Z", "Radial_Angle_Y", "Radial_Angle_X",
    "Cell_Axis_Z", "Cell_Axis_Y", "Cell_Axis_X",
    "MSD",
    "Track_Displacement", "Total_Track_Distance", "Max_Track_Distance",
    "Track_Duration",
    "Total_Intensity", "Mean_Intensity",
]


def graph_to_dataframe(
    graph: nx.DiGraph,
    *,
    seg_image: Optional[np.ndarray] = None,
    raw_image: Optional[np.ndarray] = None,
    calibration: tuple[float, float, float] = (1.0, 1.0, 1.0),
    variable_t_calibration: Optional[dict] = None,
    frame_attribute: str = "time",
    coords_attribute: str = "coords",
    label_attribute: str = "label",
) -> pd.DataFrame:
    """Produce a TrackVectors-compatible DataFrame from a Trackastra graph.

    Parameters
    ----------
    graph
        ``networkx.DiGraph`` from
        :meth:`trackastra.Trackastra.track` or any other graph that
        carries ``frame_attribute``, ``coords_attribute``, and
        ``label_attribute`` on each node.
    seg_image
        Optional ``(T, Z, Y, X)`` (or ``(T, Y, X)``) segmentation
        stack used to compute per-spot shape features
        (``Radius``, ``Eccentricity_Comp_*``, ``Cell_Axis_*``). If
        missing those columns are NaN.
    raw_image
        Same shape as ``seg_image``; enables ``Mean_Intensity`` and
        ``Total_Intensity`` columns.
    calibration
        ``(dz, dy, dx)`` voxel size — used for dynamics (Speed,
        Acceleration) and shape (Radius, Volume).
    variable_t_calibration
        Optional ``{frame: dt}`` mapping for variable timestep movies.
    frame_attribute, coords_attribute, label_attribute
        Node-attribute names to read from ``graph`` (Trackastra uses
        ``time, coords, label``; some forks use ``t``).

    Returns
    -------
    pd.DataFrame with the columns of ``TrackVectors.to_dataframe``,
    plus a ``Parent_Tracklet_Number`` column that the inverse uses
    for unambiguous lineage reconstruction.
    """
    tracklets = walk_tracklets(graph, frame_attribute=frame_attribute)

    # Number of divisions per component → Dividing / Number_Dividing.
    n_divisions_per_comp: dict[int, int] = {}
    for tl in tracklets:
        n_divisions_per_comp.setdefault(tl.component_id, 0)
    for n in graph.nodes:
        if graph.out_degree(n) >= 2:
            # find which component this node belongs to via BFS lookup
            pass
    # Simpler: count via the tracklet table — each tracklet with
    # generation > 0 has a division ancestor; total divisions in a
    # component = number of tracklets minus the number of roots minus
    # leaves-that-stem-from-a-gap. Cheap upper bound: number of
    # tracklets with generation > 0 / 2.
    # Even simpler & exact: count out_degree==2 nodes per component.
    # Build a node→component map once.
    node_to_comp: dict = {}
    for comp_id, comp in enumerate(nx.weakly_connected_components(graph)):
        for n in comp:
            node_to_comp[n] = comp_id
    for n in graph.nodes:
        if graph.out_degree(n) >= 2:
            cid = node_to_comp[n]
            n_divisions_per_comp[cid] = n_divisions_per_comp.get(cid, 0) + 1

    # Per-component aggregates: Track_Displacement / Total_Track_Distance /
    # Max_Track_Distance / Track_Duration — computed from the whole
    # component's earliest node to latest node + pairwise step sums.
    comp_aggregates: dict[int, dict] = {}
    for comp_id, comp in enumerate(nx.weakly_connected_components(graph)):
        nodes = sorted(comp, key=lambda nn: graph.nodes[nn]["__t"])
        if not nodes:
            comp_aggregates[comp_id] = {
                "displacement": 0.0, "total_distance": 0.0,
                "max_distance": 0.0, "duration": 0,
            }
            continue
        first, last = nodes[0], nodes[-1]
        p_first = np.asarray(graph.nodes[first][coords_attribute]) * np.asarray(
            calibration[: len(graph.nodes[first][coords_attribute])]
        )
        p_last = np.asarray(graph.nodes[last][coords_attribute]) * np.asarray(
            calibration[: len(graph.nodes[last][coords_attribute])]
        )
        displacement = float(np.linalg.norm(p_last - p_first))
        # Step-sum along every directed edge inside the component.
        total = 0.0
        max_step = 0.0
        for u, v in graph.edges():
            if u not in comp:
                continue
            pu = np.asarray(graph.nodes[u][coords_attribute]) * np.asarray(
                calibration[: len(graph.nodes[u][coords_attribute])]
            )
            pv = np.asarray(graph.nodes[v][coords_attribute]) * np.asarray(
                calibration[: len(graph.nodes[v][coords_attribute])]
            )
            step = float(np.linalg.norm(pv - pu))
            total += step
            if step > max_step:
                max_step = step
        comp_aggregates[comp_id] = {
            "displacement": displacement,
            "total_distance": total,
            "max_distance": max_step,
            "duration": int(graph.nodes[last]["__t"] - graph.nodes[first]["__t"]),
        }

    # Cache per-frame bboxes for shape feature lookups.
    bbox_cache_per_t: dict[int, dict[int, tuple]] = {}
    if seg_image is not None:
        for t in range(seg_image.shape[0]):
            bbox_cache_per_t[t] = _label_to_slice(seg_image[t])

    rows: list[dict] = []
    global_tracklet_id = 0
    for tl in tracklets:
        comp_id = tl.component_id
        n_div = n_divisions_per_comp.get(comp_id, 0)
        is_dividing = int(n_div > 0)
        agg = comp_aggregates[comp_id]

        # MSD per tracklet (positions in physical units).
        positions = np.asarray([
            np.asarray(graph.nodes[n][coords_attribute]) * np.asarray(
                calibration[: len(graph.nodes[n][coords_attribute])]
            )
            for n in tl.nodes
        ])
        tracklet_msd = compute_msd(positions) if len(positions) > 1 else 0.0

        prev_pos = None
        prev_prev_pos = None
        for i, n in enumerate(tl.nodes):
            data = graph.nodes[n]
            coords = np.asarray(data[coords_attribute], dtype=np.float64)
            ndim = coords.shape[0]
            cal = np.asarray(calibration[:ndim], dtype=np.float64)
            # (z, y, x) in voxel units — matches TrackVectors output.
            if ndim == 2:
                z, y, x = 0.0, float(coords[0]), float(coords[1])
            else:
                z, y, x = (float(c) for c in coords[:3])
            t_int = int(data["__t"])

            dt = compute_dt(t_int, variable_t_calibration)
            pos_phys = tuple((coords * cal).tolist())
            if ndim == 2:
                pos_phys = (0.0, *pos_phys)
            if prev_pos is not None:
                speed = compute_speed(pos_phys, prev_pos, (1.0, 1.0, 1.0)) / dt
            else:
                speed = 0.0
            if prev_prev_pos is not None:
                d2 = np.asarray(pos_phys) - 2.0 * np.asarray(prev_pos) + np.asarray(
                    prev_prev_pos
                )
                accel = float(np.linalg.norm(d2)) / (dt * dt)
            else:
                accel = 0.0
            if prev_pos is not None:
                m_z, m_y, m_x = compute_motion_angles(pos_phys, prev_pos)
            else:
                m_z, m_y, m_x = 0.0, 0.0, 0.0
            r_z, r_y, r_x = compute_radial_angles(np.asarray(pos_phys))

            row: dict = {
                "Track_ID": global_tracklet_id,
                "TrackMate_Track_ID": int(comp_id),
                "Generation_ID": int(tl.generation),
                "Tracklet_Number": int(tl.tracklet_number),
                "Parent_Tracklet_Number": (
                    int(tl.parent_tracklet_number)
                    if tl.parent_tracklet_number is not None else -1
                ),
                "t": t_int,
                "z": z, "y": y, "x": x,
                "Dividing": is_dividing,
                "Number_Dividing": int(n_div),
                "Speed": speed,
                "Motion_Angle_Z": m_z,
                "Motion_Angle_Y": m_y,
                "Motion_Angle_X": m_x,
                "Acceleration": accel,
                "Radial_Angle_Z": r_z,
                "Radial_Angle_Y": r_y,
                "Radial_Angle_X": r_x,
                "MSD": tracklet_msd,
                "Track_Displacement": agg["displacement"],
                "Total_Track_Distance": agg["total_distance"],
                "Max_Track_Distance": agg["max_distance"],
                "Track_Duration": agg["duration"],
                # Defaults — filled in by shape/intensity blocks below.
                "Radius": np.nan,
                "Eccentricity_Comp_First": np.nan,
                "Eccentricity_Comp_Second": np.nan,
                "Eccentricity_Comp_Third": np.nan,
                "Local_Cell_Density": np.nan,
                "Surface_Area": np.nan,
                "Distance_Cell_mask": np.nan,
                "Cell_Axis_Z": np.nan,
                "Cell_Axis_Y": np.nan,
                "Cell_Axis_X": np.nan,
                "Total_Intensity": np.nan,
                "Mean_Intensity": np.nan,
            }

            # Master-graph fast path: when enrich_graph_with_shape_features
            # / enrich_graph_with_dynamics were already called, the per-spot
            # features are pre-cached on the node — read them straight off,
            # no recomputation. Matches TrackVectors._master_dataframe's
            # read-from-spot.master fast path.
            _MASTER_KEYS = (
                # Shape + intensity
                "Radius", "Eccentricity_Comp_First", "Eccentricity_Comp_Second",
                "Eccentricity_Comp_Third", "Surface_Area",
                "Cell_Axis_Z", "Cell_Axis_Y", "Cell_Axis_X",
                "Total_Intensity", "Mean_Intensity",
                # Dynamics + track-level aggregates
                "Speed", "Acceleration",
                "Motion_Angle_Z", "Motion_Angle_Y", "Motion_Angle_X",
                "Radial_Angle_Z", "Radial_Angle_Y", "Radial_Angle_X",
                "MSD",
                "Track_Displacement", "Total_Track_Distance",
                "Max_Track_Distance", "Track_Duration",
            )
            for k in _MASTER_KEYS:
                if k in data:
                    row[k] = data[k]

            # Slow path: compute on the fly if the master attrs are missing.
            if seg_image is not None and label_attribute in data and (
                "Radius" not in data or np.isnan(data.get("Radius", np.nan))
            ):
                label = int(data[label_attribute])
                bcache = bbox_cache_per_t.get(t_int, {})
                raw_frame = (
                    raw_image[t_int]
                    if raw_image is not None and t_int < raw_image.shape[0]
                    else None
                )
                row.update(_shape_props_for(
                    seg_image[t_int], label,
                    bbox_cache=bcache,
                    calibration=calibration,
                    raw_frame=raw_frame,
                ))

            rows.append(row)
            prev_prev_pos = prev_pos
            prev_pos = pos_phys

        global_tracklet_id += 1

    df = pd.DataFrame(rows, columns=_COLUMNS)
    return df


# ============================================================ reverse

def dataframe_to_graph(
    df: pd.DataFrame,
    *,
    coords_attribute: str = "coords",
    frame_attribute: str = "time",
    label_attribute: str = "label",
) -> nx.DiGraph:
    """Round-trip: turn a (possibly Oneat-corrected) DataFrame back into a
    Trackastra-shaped ``networkx.DiGraph``.

    Reconstruction rules — match :func:`graph_to_dataframe`:

    - Each ``Track_ID`` group → one linear chain of edges, ordered by ``t``.
    - At every division boundary (same ``TrackMate_Track_ID``, child's
      ``Generation_ID == parent.Generation_ID + 1``, child's
      ``Parent_Tracklet_Number == parent.Tracklet_Number``), add an
      edge from the parent tracklet's last node to the child tracklet's
      first node.
    - Node id = ``(t, Track_ID, sample_index_within_tracklet)`` — keeps
      ids unique without depending on a Spot_ID column.

    Returns a ``nx.DiGraph`` with the same node attribute layout
    (``time``, ``coords``, ``label``, ``weight=1``) Trackastra
    produces, so downstream Trackastra utilities (greedy/ILP refits,
    ``apply_solution_graph_to_masks``, the napari plugin) keep working.
    """
    if df.empty:
        return nx.DiGraph()
    G = nx.DiGraph()

    # Stable per-tracklet ordering on (t, original row index).
    df = df.sort_values(["TrackMate_Track_ID", "Track_ID", "t"])

    # First pass — emit nodes + intra-tracklet edges.
    tracklet_first_node: dict[int, object] = {}
    tracklet_last_node: dict[int, object] = {}
    for tid, sub in df.groupby("Track_ID", sort=False):
        sub = sub.reset_index(drop=True)
        prev_node = None
        for i, row in sub.iterrows():
            t = int(row["t"])
            coords = (float(row["z"]), float(row["y"]), float(row["x"]))
            node_id = (t, int(tid), int(i))
            attrs = {
                frame_attribute: t,
                "t": t,
                coords_attribute: coords,
                "weight": 1.0,
            }
            if label_attribute in df.columns:
                attrs[label_attribute] = int(row[label_attribute])
            G.add_node(node_id, **attrs)
            if prev_node is not None:
                G.add_edge(prev_node, node_id, weight=1.0)
            prev_node = node_id
            if i == 0:
                tracklet_first_node[int(tid)] = node_id
            tracklet_last_node[int(tid)] = node_id

    # Second pass — emit division edges from Parent_Tracklet_Number.
    if "Parent_Tracklet_Number" in df.columns:
        # Map (component_id, tracklet_number) → Track_ID for parent lookup.
        comp_tnum_to_tid: dict[tuple[int, int], int] = {}
        for tid, sub in df.groupby("Track_ID", sort=False):
            comp = int(sub["TrackMate_Track_ID"].iloc[0])
            tnum = int(sub["Tracklet_Number"].iloc[0])
            comp_tnum_to_tid[(comp, tnum)] = int(tid)

        for tid, sub in df.groupby("Track_ID", sort=False):
            parent_tnum = int(sub["Parent_Tracklet_Number"].iloc[0])
            if parent_tnum < 0:
                continue
            comp = int(sub["TrackMate_Track_ID"].iloc[0])
            parent_tid = comp_tnum_to_tid.get((comp, parent_tnum))
            if parent_tid is None:
                continue
            parent_last = tracklet_last_node.get(parent_tid)
            child_first = tracklet_first_node.get(int(tid))
            if parent_last is None or child_first is None:
                continue
            G.add_edge(parent_last, child_first, weight=1.0)
    return G

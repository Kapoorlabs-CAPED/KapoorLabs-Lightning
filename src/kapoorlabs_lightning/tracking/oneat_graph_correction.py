"""Apply Oneat-predicted division events to a Trackastra DiGraph.

Mirrors what TrackMate-Oneat does in the Fiji ecosystem — except the
edits happen on the in-memory ``networkx.DiGraph`` produced by
Trackastra rather than a TrackMate XML. The corrected graph then
flows through :func:`enrich_graph_with_shape_features` (per-spot
shape + intensity cached as node attrs — the "master corrected
graph"), and finally through
:func:`kapoorlabs_lightning.tracking.trackastra_bridge.graph_to_dataframe`
to land in the same DataFrame schema TrackMate-corrected data does.

Oneat CSV schema (written by
``KapoorLabs-Lightning/scripts/model_prediction/predict-oneat.py``)::

    t, z, y, x, score, size, h, w, d        (event_name baked into filename)

The correction logic per event is:

1. Match the event ``(t, z, y, x)`` to the **nearest existing node**
   in frame ``t`` by centroid Euclidean distance, gated by
   ``max_match_distance`` in physical units.
2. If the matched node already has ``out_degree >= 2``, skip — the
   division is already modelled.
3. Else look in frame ``t + 1`` (extended to ``t + max_gap``) for
   candidate daughter cells within ``max_daughter_distance``:
   - parent with 0 successors → add the 2 nearest candidates.
   - parent with 1 successor   → add the 1 nearest *additional*
     candidate (not the existing successor).
4. Edges added carry ``weight=oneat_score`` so downstream tooling
   can tell which links came from Oneat correction.
5. The parent node gets ``division_corrected=True`` for audit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ============================================================ CSV loading

def load_oneat_events(
    source: Union[str, Path, pd.DataFrame, Iterable[dict]],
    *,
    event_name: Optional[str] = None,
    score_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Normalise an Oneat event source into a DataFrame with ``t,z,y,x[,score]``.

    Accepts a CSV path, a directory of Oneat CSVs (concatenated), a
    DataFrame, or any iterable of dicts. The ``event_name`` filter is
    matched against either a ``event_name`` column or — for the
    per-event CSV files Oneat writes — the file stem
    (``oneat_Division_<basename>.csv``).
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            frames = []
            for f in sorted(p.glob("*.csv")):
                stem = f.stem
                if event_name and event_name.lower() not in stem.lower():
                    continue
                tmp = pd.read_csv(f)
                tmp.setdefault("event_name", stem.split("_")[1] if "_" in stem else stem)
                frames.append(tmp)
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            df = pd.read_csv(p)
            if "event_name" not in df.columns:
                df["event_name"] = p.stem.split("_")[1] if "_" in p.stem else p.stem
    else:
        df = pd.DataFrame(list(source))

    if df.empty:
        return df

    if event_name and "event_name" in df.columns:
        df = df[df["event_name"].astype(str).str.lower() == event_name.lower()]
    if score_threshold is not None and "score" in df.columns:
        df = df[df["score"] >= float(score_threshold)]

    # Be tolerant of older / TrackMate-style column names.
    rename = {"T": "t", "Z": "z", "Y": "y", "X": "x", "time": "t"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    required = {"t", "z", "y", "x"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Oneat CSV is missing required columns: {sorted(missing)}")
    df = df.reset_index(drop=True)
    return df


# ============================================================ graph indexing

def _frame_kdtree(
    graph: nx.DiGraph,
    *,
    frame_attribute: str,
    coords_attribute: str,
    calibration: tuple[float, ...],
) -> tuple[dict[int, cKDTree], dict[int, list]]:
    """Per-frame KDTree of node centroids (physical units) for fast NN lookup."""
    by_frame: dict[int, list] = {}
    cal = np.asarray(calibration, dtype=np.float64)
    for n, data in graph.nodes(data=True):
        t = int(data.get(frame_attribute, data.get("t", data.get("time"))))
        by_frame.setdefault(t, []).append(n)

    trees: dict[int, cKDTree] = {}
    ordered: dict[int, list] = {}
    for t, nodes in by_frame.items():
        pts = []
        for n in nodes:
            coords = np.asarray(graph.nodes[n][coords_attribute], dtype=np.float64)
            pts.append((coords * cal[: coords.shape[0]])[: coords.shape[0]])
        # Pad 2D to 3D so all queries are 3D regardless of source dim.
        pts3 = []
        for p in pts:
            if p.shape[0] == 2:
                pts3.append(np.asarray([0.0, p[0], p[1]]))
            else:
                pts3.append(p[:3])
        trees[t] = cKDTree(np.stack(pts3))
        ordered[t] = list(nodes)
    return trees, ordered


# ============================================================ correction

def oneat_correct_graph(
    graph: nx.DiGraph,
    events: Union[str, Path, pd.DataFrame, Iterable[dict]],
    *,
    event_name: Optional[str] = "Division",
    score_threshold: Optional[float] = None,
    calibration: tuple[float, ...] = (1.0, 1.0, 1.0),
    max_match_distance: float = 20.0,
    max_daughter_distance: float = 30.0,
    max_gap: int = 1,
    coords_attribute: str = "coords",
    frame_attribute: str = "time",
) -> tuple[nx.DiGraph, pd.DataFrame]:
    """Insert missed-division edges from Oneat events into ``graph``.

    Parameters
    ----------
    graph
        Trackastra-shaped ``DiGraph`` (nodes with ``frame_attribute``
        and ``coords_attribute``). Operated on a copy — the input is
        not mutated.
    events
        Oneat CSV path / directory / DataFrame / list of dicts (see
        :func:`load_oneat_events`).
    event_name
        Filter the input to division events (default ``"Division"``).
        Pass ``None`` to accept everything.
    score_threshold
        Optional minimum Oneat ``score`` to act on.
    calibration
        ``(dz, dy, dx)`` voxel size — matching distances are
        computed in physical units.
    max_match_distance
        Max distance between the event centroid and the candidate
        parent node, both in frame ``t``.
    max_daughter_distance
        Max distance between the parent and a candidate daughter
        in frame ``t + 1`` (or up to ``t + max_gap``).
    max_gap
        How many frames forward to scan for daughters when no
        immediate next-frame candidate is found.

    Returns
    -------
    (corrected_graph, audit_df)
        ``corrected_graph`` is a copy of ``graph`` with the new
        division edges added. ``audit_df`` has one row per event
        and columns
        ``t, x_event, y_event, z_event, score, matched_node,
        action, n_daughters_added, distance, notes``.
    """
    df_events = load_oneat_events(
        events, event_name=event_name, score_threshold=score_threshold,
    )
    if df_events.empty:
        return graph.copy(), pd.DataFrame()

    G = graph.copy()
    trees, ordered = _frame_kdtree(
        G, frame_attribute=frame_attribute,
        coords_attribute=coords_attribute, calibration=calibration,
    )

    audit_rows: list[dict] = []

    cal = np.asarray(calibration, dtype=np.float64)
    cal3 = np.asarray([cal[0] if cal.size >= 1 else 1.0,
                       cal[1] if cal.size >= 2 else cal[-1],
                       cal[2] if cal.size >= 3 else cal[-1]], dtype=np.float64)

    for _, ev in df_events.iterrows():
        t = int(ev["t"])
        ev_pt = np.asarray([
            float(ev["z"]) * cal3[0],
            float(ev["y"]) * cal3[1],
            float(ev["x"]) * cal3[2],
        ])
        score = float(ev.get("score", 1.0))

        audit = {
            "t": t,
            "z_event": float(ev["z"]),
            "y_event": float(ev["y"]),
            "x_event": float(ev["x"]),
            "score": score,
            "matched_node": None,
            "action": "no_match",
            "n_daughters_added": 0,
            "distance": float("nan"),
            "notes": "",
        }

        tree_t = trees.get(t)
        if tree_t is None:
            audit["notes"] = f"no nodes in frame {t}"
            audit_rows.append(audit); continue

        d, idx = tree_t.query(ev_pt, k=1)
        if d > max_match_distance:
            audit["distance"] = float(d)
            audit["notes"] = (
                f"nearest node at d={d:.2f} > max_match_distance="
                f"{max_match_distance}"
            )
            audit_rows.append(audit); continue

        parent = ordered[t][int(idx)]
        audit["matched_node"] = parent
        audit["distance"] = float(d)

        if G.out_degree(parent) >= 2:
            audit["action"] = "already_dividing"
            audit_rows.append(audit); continue

        existing_succ = set(G.successors(parent))
        need = 2 - len(existing_succ)
        if need <= 0:
            audit["action"] = "already_dividing"
            audit_rows.append(audit); continue

        parent_pt = np.asarray(
            G.nodes[parent][coords_attribute], dtype=np.float64,
        )
        if parent_pt.shape[0] == 2:
            parent_pt = np.asarray([0.0, parent_pt[0], parent_pt[1]])
        parent_pt = parent_pt[:3] * cal3[: parent_pt.shape[0]]

        # Look for daughters in t+1 .. t+max_gap.
        candidates: list[tuple[float, object]] = []
        for dt in range(1, max_gap + 1):
            tree_n = trees.get(t + dt)
            if tree_n is None:
                continue
            dists, idxs = tree_n.query(parent_pt, k=min(8, len(ordered[t + dt])))
            dists = np.atleast_1d(dists)
            idxs = np.atleast_1d(idxs)
            for dd, ii in zip(dists.tolist(), idxs.tolist()):
                if dd > max_daughter_distance:
                    continue
                node = ordered[t + dt][int(ii)]
                if node in existing_succ:
                    continue
                # Don't steal a node that already has a parent.
                if G.in_degree(node) >= 1:
                    continue
                candidates.append((float(dd), node))
        candidates.sort(key=lambda x: x[0])

        if not candidates:
            audit["action"] = "no_daughter_candidate"
            audit["notes"] = "no eligible daughters within max_daughter_distance"
            audit_rows.append(audit); continue

        picked = candidates[:need]
        for dd, dnode in picked:
            # Avoid edge-attribute key "source" — clashes with the
            # networkx node-link JSON convention used by
            # ``write_master_graph``.
            G.add_edge(parent, dnode, weight=score, edit_source="oneat")
        G.nodes[parent]["division_corrected"] = True
        audit["action"] = (
            "added_division" if (len(existing_succ) + len(picked) == 2)
            else "added_partial_division"
        )
        audit["n_daughters_added"] = len(picked)
        audit_rows.append(audit)

    audit_df = pd.DataFrame(audit_rows)
    return G, audit_df

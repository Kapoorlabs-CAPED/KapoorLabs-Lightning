"""
Track comparison metrics.

Compares ground-truth and predicted TrackMate XMLs using optimal
assignment, computing CCA (Cell Cycle Accuracy), CT (Complete Tracks),
and BCI (Branching Correctness Index) metrics.
"""

import os
import numpy as np
import pandas as pd
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Union

from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

from .xml_parser import TrackMateXML
from .track_vectors import TrackVectors


class TrackComparator:
    """
    Compare ground-truth and predicted tracks via optimal assignment.

    Builds a cost matrix using symmetric mean KDTree distances between
    track point clouds, then uses the Hungarian algorithm for assignment.

    Args:
        gt: Path to GT TrackMate XML or pre-parsed TrackMateXML.
        pred: Path to predicted TrackMate XML or pre-parsed TrackMateXML.
        downsampleT: Temporal downsampling factor for predicted tracks.
            Used to align predicted frame indices back to GT grid.

    Example:
        >>> comp = TrackComparator("gt_tracks.xml", "pred_tracks.xml")
        >>> results = comp.evaluate(threshold=10.0)
        >>> print(f"Hits: {results['num_hits']}/{results['num_gt']}")
    """

    def __init__(
        self,
        gt: Union[str, TrackMateXML],
        pred: Union[str, TrackMateXML],
        downsampleT: int = 1,
    ):
        self.gt_xml = TrackMateXML(gt) if isinstance(gt, str) else gt
        self.pred_xml = TrackMateXML(pred) if isinstance(pred, str) else pred

        self.gt_df = self._build_track_df(self.gt_xml)
        self.pred_df = self._build_track_df(self.pred_xml)

        raw_gt = self._track_clouds(self.gt_df)
        raw_pred = self._track_clouds(self.pred_df)

        self.downsampleT = max(1, downsampleT)
        self.gt_items = list(raw_gt.items())
        self.pred_items = list(raw_pred.items())

    @staticmethod
    def _build_track_df(xml: TrackMateXML) -> pd.DataFrame:
        """Build a minimal DataFrame of (track_id, t, z, y, x) from parsed XML."""
        rows = []
        for track_id in xml.filtered_track_ids:
            spot_ids = xml.get_all_spot_ids_for_track(track_id)
            for sid in spot_ids:
                spot = xml.spots[sid]
                rows.append({
                    "track_id": track_id,
                    "t": spot.frame,
                    "z": spot.z,
                    "y": spot.y,
                    "x": spot.x,
                })
        return pd.DataFrame(rows)

    @staticmethod
    def _track_clouds(df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """Group spots by track_id into Nx3 arrays."""
        return {
            tid: grp[["z", "y", "x"]].values
            for tid, grp in df.groupby("track_id")
        }

    def evaluate(
        self,
        threshold: float,
        compute_bci: bool = False,
    ) -> Dict[str, object]:
        """
        Run optimal assignment and compute metrics.

        Args:
            threshold: Maximum distance for a match.
            compute_bci: Whether to compute branching correctness index.

        Returns:
            Dict with keys: assignments (DataFrame), num_hits, num_gt,
            num_pred, cca, ct, bci (if requested).
        """
        num_gt = len(self.gt_items)
        num_pred = len(self.pred_items)

        if num_gt == 0 or num_pred == 0:
            return {
                "assignments": pd.DataFrame(),
                "num_hits": 0,
                "num_gt": num_gt,
                "num_pred": num_pred,
                "cca": np.nan,
                "ct": np.nan,
                "bci": np.nan,
            }

        # Pre-build KDTrees for predicted tracks
        pred_trees = {}
        for pid, coords in self.pred_items:
            if coords.size:
                pred_trees[pid] = cKDTree(coords)

        def compute_row(item):
            i, (gt_id, gt_coords) = item
            row = np.full(num_pred, np.inf)
            if gt_coords.size:
                tree_gt = cKDTree(gt_coords)
                for j, (pred_id, pred_coords) in enumerate(self.pred_items):
                    if pred_id in pred_trees:
                        d_gt = pred_trees[pred_id].query(gt_coords)[0]
                        d_pred = tree_gt.query(pred_coords)[0]
                        row[j] = max(d_gt.mean(), d_pred.mean())
            return i, row

        # Build cost matrix with parallel computation
        cost = np.full((num_gt, num_pred), np.inf)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            futures = [
                executor.submit(compute_row, item)
                for item in enumerate(self.gt_items)
            ]
            for f in concurrent.futures.as_completed(futures):
                i, row = f.result()
                cost[i] = row

        # Hungarian assignment
        gt_idx, pred_idx = linear_sum_assignment(cost)

        records = []
        for i, j in zip(gt_idx, pred_idx):
            gt_id = self.gt_items[i][0]
            pred_id = self.pred_items[j][0]
            dist = float(cost[i, j])
            records.append({
                "gt_track": gt_id,
                "pred_track": pred_id,
                "distance": dist,
                "matched": dist <= threshold,
            })

        assignments = pd.DataFrame(records)
        num_hits = int(assignments["matched"].sum())

        cca = self.cca_metric()
        ct = self.ct_metric(assignments)
        bci = self.bci_metric(assignments) if compute_bci else None

        return {
            "assignments": assignments,
            "num_hits": num_hits,
            "num_gt": num_gt,
            "num_pred": num_pred,
            "cca": cca,
            "ct": ct,
            "bci": bci,
        }

    def cca_metric(self) -> float:
        """
        Cell Cycle Accuracy: 1 - max CDF distance of track-length histograms.

        Compares the distribution of track durations between GT and predicted.
        """
        gt_lengths = np.array([
            self.gt_df[self.gt_df["track_id"] == tid]["t"].max()
            - self.gt_df[self.gt_df["track_id"] == tid]["t"].min()
            for tid, _ in self.gt_items
        ], dtype=int)

        pred_lengths = np.array([
            (self.pred_df[self.pred_df["track_id"] == tid]["t"].max()
             - self.pred_df[self.pred_df["track_id"] == tid]["t"].min())
            * self.downsampleT
            for tid, _ in self.pred_items
        ], dtype=int)

        if gt_lengths.size == 0:
            return np.nan

        M = max(gt_lengths.max(), pred_lengths.max())
        h_gt = np.bincount(gt_lengths, minlength=M + 1).astype(float)
        h_gt /= h_gt.sum()
        c_gt = np.cumsum(h_gt)

        h_pred = np.bincount(pred_lengths, minlength=M + 1).astype(float)
        h_pred /= h_pred.sum()
        c_pred = np.cumsum(h_pred)

        return float(1 - np.max(np.abs(c_gt - c_pred)))

    def ct_metric(self, assignments: pd.DataFrame) -> float:
        """
        Complete Tracks: fraction with matching start/end frames.

        Measures how many assigned track pairs have identical temporal spans.
        """
        T_r = len(self.gt_items)
        T_c = len(self.pred_items)

        if (T_r + T_c) == 0:
            return np.nan

        gt_spans = {}
        for tid, _ in self.gt_items:
            sub = self.gt_df[self.gt_df["track_id"] == tid]
            gt_spans[tid] = (int(sub["t"].min()), int(sub["t"].max()))

        pred_spans = {}
        for tid, _ in self.pred_items:
            sub = self.pred_df[self.pred_df["track_id"] == tid]
            b = int(sub["t"].min()) * self.downsampleT
            e = int(sub["t"].max()) * self.downsampleT
            pred_spans[tid] = (b, e)

        T_rc = 0
        for _, row in assignments.iterrows():
            gt_id = row["gt_track"]
            pred_id = row["pred_track"]
            if gt_spans.get(gt_id) == pred_spans.get(pred_id):
                T_rc += 1

        return float(2 * T_rc / (T_r + T_c))

    def _get_mitosis_events(self, xml: TrackMateXML) -> List[Tuple[int, int]]:
        """
        Get all mitosis (split) events as (track_id, split_frame) tuples.
        """
        events = []
        for track_id in xml.filtered_track_ids:
            if not xml.is_dividing(track_id):
                continue
            for split_spot_id in xml.get_split_ids(track_id):
                spot = xml.spots.get(split_spot_id)
                if spot is not None:
                    events.append((track_id, spot.frame))
        return events

    def bci_metric(
        self,
        assignments: pd.DataFrame,
        tol: int = 1,
    ) -> float:
        """
        Branching Correctness Index (F1 score of mitotic events).

        Matches GT split events to predicted split events within
        assigned track pairs and a temporal tolerance.

        Args:
            assignments: Assignment DataFrame from evaluate().
            tol: Frame tolerance for matching split events.

        Returns:
            F1 score of mitotic event detection.
        """
        gt_events = self._get_mitosis_events(self.gt_xml)
        pred_events = self._get_mitosis_events(self.pred_xml)

        # Map GT track → assigned pred track (matched only)
        assign_map = {
            int(r["gt_track"]): int(r["pred_track"])
            for _, r in assignments.iterrows()
            if r["matched"]
        }

        tp = fp = fn = 0

        # True positives and false negatives
        for gt_tid, t_gt in gt_events:
            pr_tid = assign_map.get(gt_tid)
            if pr_tid is None:
                fn += 1
            elif any(
                pr == pr_tid and abs(t_pr - t_gt) <= tol
                for pr, t_pr in pred_events
            ):
                tp += 1
            else:
                fn += 1

        # False positives
        matched_preds = set(assign_map.values())
        for pr_tid, t_pr in pred_events:
            if pr_tid not in matched_preds:
                fp += 1
            else:
                gt_tids = [gt for gt, pr in assign_map.items() if pr == pr_tid]
                matched_any = any(
                    gt2 == candidate_gt and abs(t_pr - t_gt2) <= tol
                    for candidate_gt in gt_tids
                    for gt2, t_gt2 in gt_events
                )
                if not matched_any:
                    fp += 1

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return float(2 * precision * recall / max(precision + recall, 1e-8))

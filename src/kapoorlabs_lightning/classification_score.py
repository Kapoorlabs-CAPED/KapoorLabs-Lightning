"""
Classification scoring for ONEAT event detection.
Compares predicted events against ground truth annotations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict, Tuple
from scipy.spatial.distance import cdist


class ClassificationScore:
    """
    Compute classification metrics for event detection.

    Compares predictions CSV (all events) against ground truth CSV (single event type)
    using spatial-temporal matching with configurable thresholds.

    Args:
        predictions_csv: Path to predictions CSV with columns [t, z, y, x, event_name/class]
        ground_truth_csv: Path to ground truth CSV with columns [t, z, y, x]
        event_name: Name of the event to evaluate (e.g., 'mitosis', 'apoptosis')
        match_threshold_space: Spatial distance threshold for matching (default: 10)
        match_threshold_time: Temporal distance threshold for matching (default: 2)
        prediction_class_column: Column name for predicted class (default: 'event_name')
    """

    def __init__(
        self,
        predictions_csv: Union[str, Path],
        ground_truth_csv: Union[str, Path],
        event_name: str,
        match_threshold_space: float = 10.0,
        match_threshold_time: int = 2,
        prediction_class_column: str = 'event_name',
    ):
        self.predictions_csv = Path(predictions_csv)
        self.ground_truth_csv = Path(ground_truth_csv)
        self.event_name = event_name
        self.match_threshold_space = match_threshold_space
        self.match_threshold_time = match_threshold_time
        self.prediction_class_column = prediction_class_column

        # Load and process data
        self.predictions_df = self._load_predictions()
        self.ground_truth_df = self._load_ground_truth()

        # Compute matches
        self._compute_matches()

    def _load_predictions(self) -> pd.DataFrame:
        """Load predictions CSV and filter for the target event."""
        df = pd.read_csv(self.predictions_csv)
        df.columns = [col.lower() for col in df.columns]

        # Filter for target event
        class_col = self.prediction_class_column.lower()
        if class_col in df.columns:
            df_filtered = df[df[class_col] == self.event_name].copy()
        else:
            # Try common column names
            for col in ['event_name', 'class', 'predicted_class', 'label']:
                if col in df.columns:
                    df_filtered = df[df[col] == self.event_name].copy()
                    break
            else:
                raise ValueError(f"Could not find class column in predictions. Columns: {df.columns.tolist()}")

        print(f"Loaded {len(df_filtered)} predictions for '{self.event_name}' (total: {len(df)})")
        return df_filtered

    def _load_ground_truth(self) -> pd.DataFrame:
        """Load ground truth CSV."""
        df = pd.read_csv(self.ground_truth_csv)
        df.columns = [col.lower() for col in df.columns]
        print(f"Loaded {len(df)} ground truth events for '{self.event_name}'")
        return df

    def _get_coordinates(self, df: pd.DataFrame) -> np.ndarray:
        """Extract TZYX coordinates from dataframe."""
        coords = []
        for _, row in df.iterrows():
            t = row.get('t', row.get('time', 0))
            z = row.get('z', 0)
            y = row.get('y', 0)
            x = row.get('x', 0)
            coords.append([t, z, y, x])
        return np.array(coords)

    def _compute_matches(self):
        """Compute matches between predictions and ground truth."""
        if len(self.predictions_df) == 0 or len(self.ground_truth_df) == 0:
            self.true_positives = 0
            self.false_positives = len(self.predictions_df)
            self.false_negatives = len(self.ground_truth_df)
            self.matched_predictions = []
            self.matched_ground_truth = []
            self.unmatched_predictions = list(range(len(self.predictions_df)))
            self.unmatched_ground_truth = list(range(len(self.ground_truth_df)))
            return

        pred_coords = self._get_coordinates(self.predictions_df)
        gt_coords = self._get_coordinates(self.ground_truth_df)

        # Compute pairwise distances
        # Separate time and space for different thresholds
        pred_time = pred_coords[:, 0:1]
        pred_space = pred_coords[:, 1:]
        gt_time = gt_coords[:, 0:1]
        gt_space = gt_coords[:, 1:]

        # Time distance (L1)
        time_dist = cdist(pred_time, gt_time, metric='cityblock')

        # Spatial distance (Euclidean)
        space_dist = cdist(pred_space, gt_space, metric='euclidean')

        # Valid matches: within both thresholds
        valid_matches = (time_dist <= self.match_threshold_time) & \
                        (space_dist <= self.match_threshold_space)

        # Greedy matching: assign each GT to closest valid prediction
        matched_pred_indices = set()
        matched_gt_indices = set()
        matches = []

        # Combined distance for ranking
        combined_dist = time_dist + space_dist
        combined_dist[~valid_matches] = np.inf

        # Sort by distance and greedily match
        while True:
            if np.all(np.isinf(combined_dist)):
                break

            # Find minimum distance
            min_idx = np.unravel_index(np.argmin(combined_dist), combined_dist.shape)
            pred_idx, gt_idx = min_idx

            if combined_dist[pred_idx, gt_idx] == np.inf:
                break

            # Record match
            matches.append((pred_idx, gt_idx))
            matched_pred_indices.add(pred_idx)
            matched_gt_indices.add(gt_idx)

            # Remove matched from consideration
            combined_dist[pred_idx, :] = np.inf
            combined_dist[:, gt_idx] = np.inf

        self.true_positives = len(matches)
        self.false_positives = len(self.predictions_df) - len(matched_pred_indices)
        self.false_negatives = len(self.ground_truth_df) - len(matched_gt_indices)

        self.matched_predictions = list(matched_pred_indices)
        self.matched_ground_truth = list(matched_gt_indices)
        self.unmatched_predictions = [i for i in range(len(self.predictions_df)) if i not in matched_pred_indices]
        self.unmatched_ground_truth = [i for i in range(len(self.ground_truth_df)) if i not in matched_gt_indices]
        self.matches = matches

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        """F1 = 2 * (Precision * Recall) / (Precision + Recall)"""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def accuracy(self) -> float:
        """Accuracy = TP / (TP + FP + FN)"""
        total = self.true_positives + self.false_positives + self.false_negatives
        if total == 0:
            return 0.0
        return self.true_positives / total

    def get_metrics(self) -> Dict[str, float]:
        """Get all metrics as a dictionary."""
        return {
            'event_name': self.event_name,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'accuracy': self.accuracy,
            'n_predictions': len(self.predictions_df),
            'n_ground_truth': len(self.ground_truth_df),
        }

    def print_report(self):
        """Print a formatted classification report."""
        metrics = self.get_metrics()

        print(f"\n{'='*60}")
        print(f"Classification Report: {self.event_name}")
        print(f"{'='*60}")
        print(f"Match thresholds: space={self.match_threshold_space}, time={self.match_threshold_time}")
        print(f"-"*60)
        print(f"Ground Truth Events:  {metrics['n_ground_truth']}")
        print(f"Predicted Events:     {metrics['n_predictions']}")
        print(f"-"*60)
        print(f"True Positives:       {metrics['true_positives']}")
        print(f"False Positives:      {metrics['false_positives']}")
        print(f"False Negatives:      {metrics['false_negatives']}")
        print(f"-"*60)
        print(f"Precision:            {metrics['precision']:.4f}")
        print(f"Recall:               {metrics['recall']:.4f}")
        print(f"F1 Score:             {metrics['f1_score']:.4f}")
        print(f"Accuracy:             {metrics['accuracy']:.4f}")
        print(f"{'='*60}\n")

    def get_false_positives_df(self) -> pd.DataFrame:
        """Get dataframe of false positive predictions."""
        if len(self.unmatched_predictions) == 0:
            return pd.DataFrame()
        return self.predictions_df.iloc[self.unmatched_predictions].copy()

    def get_false_negatives_df(self) -> pd.DataFrame:
        """Get dataframe of missed ground truth events."""
        if len(self.unmatched_ground_truth) == 0:
            return pd.DataFrame()
        return self.ground_truth_df.iloc[self.unmatched_ground_truth].copy()

    def save_report(self, output_path: Union[str, Path]):
        """Save metrics to a CSV file."""
        metrics = self.get_metrics()
        df = pd.DataFrame([metrics])
        df.to_csv(output_path, index=False)
        print(f"Report saved to {output_path}")


def evaluate_multiple_events(
    predictions_csv: Union[str, Path],
    ground_truth_dir: Union[str, Path],
    event_names: list,
    match_threshold_space: float = 10.0,
    match_threshold_time: int = 2,
    gt_filename_pattern: str = "oneat_{event_name}_*.csv",
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple event types.

    Args:
        predictions_csv: Path to predictions CSV
        ground_truth_dir: Directory containing ground truth CSVs
        event_names: List of event names to evaluate
        match_threshold_space: Spatial matching threshold
        match_threshold_time: Temporal matching threshold
        gt_filename_pattern: Pattern for GT files, {event_name} will be replaced

    Returns:
        Dictionary of event_name -> metrics
    """
    ground_truth_dir = Path(ground_truth_dir)
    results = {}

    for event_name in event_names:
        # Find ground truth file for this event
        pattern = gt_filename_pattern.replace("{event_name}", event_name)
        gt_files = list(ground_truth_dir.glob(pattern))

        if len(gt_files) == 0:
            print(f"Warning: No ground truth file found for '{event_name}'")
            continue

        # If multiple files, concatenate them
        if len(gt_files) > 1:
            dfs = [pd.read_csv(f) for f in gt_files]
            combined_gt = pd.concat(dfs, ignore_index=True)
            # Save to temp file
            temp_gt = ground_truth_dir / f"_temp_combined_{event_name}.csv"
            combined_gt.to_csv(temp_gt, index=False)
            gt_file = temp_gt
        else:
            gt_file = gt_files[0]

        scorer = ClassificationScore(
            predictions_csv=predictions_csv,
            ground_truth_csv=gt_file,
            event_name=event_name,
            match_threshold_space=match_threshold_space,
            match_threshold_time=match_threshold_time,
        )

        scorer.print_report()
        results[event_name] = scorer.get_metrics()

    return results

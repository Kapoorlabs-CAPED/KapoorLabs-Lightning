#!/usr/bin/env python3
"""
Compute classification accuracy for ONEAT event predictions.
"""

import os
from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from kapoorlabs_lightning import ClassificationScore
from scenario_compute_accuracy import AccuracyConfig


configstore = ConfigStore.instance()
configstore.store(name="AccuracyConfig", node=AccuracyConfig)


@hydra.main(
    config_path="../conf", config_name="scenario_compute_accuracy"
)
def main(config: AccuracyConfig):

    predictions_csv = config.paths.predictions_csv
    ground_truth_csv = config.paths.ground_truth_csv
    output_dir = config.paths.output_dir

    event_name = config.params.event_name
    match_threshold_space = config.params.match_threshold_space
    match_threshold_time = config.params.match_threshold_time
    prediction_class_column = config.params.prediction_class_column
    save_errors = config.params.save_errors

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Compute scores
    scorer = ClassificationScore(
        predictions_csv=predictions_csv,
        ground_truth_csv=ground_truth_csv,
        event_name=event_name,
        match_threshold_space=match_threshold_space,
        match_threshold_time=match_threshold_time,
        prediction_class_column=prediction_class_column,
    )

    # Print report
    scorer.print_report()

    # Save metrics
    metrics_path = os.path.join(output_dir, f"metrics_{event_name}.csv")
    scorer.save_report(metrics_path)

    # Save error analysis if requested
    if save_errors:
        fp_df = scorer.get_false_positives_df()
        if len(fp_df) > 0:
            fp_path = os.path.join(output_dir, f"false_positives_{event_name}.csv")
            fp_df.to_csv(fp_path, index=False)
            print(f"False positives saved to: {fp_path}")

        fn_df = scorer.get_false_negatives_df()
        if len(fn_df) > 0:
            fn_path = os.path.join(output_dir, f"false_negatives_{event_name}.csv")
            fn_df.to_csv(fn_path, index=False)
            print(f"False negatives saved to: {fn_path}")


if __name__ == "__main__":
    main()

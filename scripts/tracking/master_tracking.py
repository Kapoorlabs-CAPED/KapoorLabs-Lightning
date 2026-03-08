#!/usr/bin/env python3
"""
Master tracking: enhance TrackMate XML with morphological shape features.

Reads a TrackMate XML and its corresponding segmentation image, computes
point cloud shape features (eccentricity, surface area, cell axis angles)
for each segmented cell, matches them to XML spots via spatial proximity,
and writes an enhanced "master" XML with these features embedded.

Optionally also:
  - Saves the full feature DataFrame as CSV (normalized or raw)
  - Transfers tracks from one channel to another (e.g. nuclei → membrane)

Usage:
    python master_tracking.py
    python master_tracking.py train_data_paths=tracking_default
    python master_tracking.py parameters.do_nuclei=true parameters.do_membrane=true
    python master_tracking.py parameters.do_channel_transfer=true
"""

import os
import hydra
import numpy as np
from pathlib import Path
from tifffile import imread
from hydra.core.config_store import ConfigStore

from kapoorlabs_lightning.tracking.track_vectors import TrackVectors
from kapoorlabs_lightning.tracking.xml_writer import write_trackmate_xml
from kapoorlabs_lightning.tracking.channel_transfer import ChannelTransfer
from kapoorlabs_lightning.tracking.track_features import SHAPE_DYNAMIC_FEATURES
from scenario_master_tracking import MasterTrackingClass


configstore = ConfigStore.instance()
configstore.store(name="MasterTrackingClass", node=MasterTrackingClass)


def normalize_features(df, feature_columns):
    """Per-column z-score normalization."""
    df_norm = df.copy()
    for col in feature_columns:
        if col in df_norm.columns:
            mean = df_norm[col].mean()
            std = df_norm[col].std()
            if std > 0:
                df_norm[col] = (df_norm[col] - mean) / std
            else:
                df_norm[col] = 0.0
    return df_norm


def build_spot_properties(df, xml):
    """
    Build a spot properties dict from the feature DataFrame.

    Maps each spot (by matching Track_ID + t + position) to its computed
    shape features, formatted as XML attribute names.
    """
    spot_properties = {}

    for track_id in xml.filtered_track_ids:
        spot_ids = xml.get_all_spot_ids_for_track(track_id)

        for sid in spot_ids:
            spot = xml.spots[sid]

            # Find matching row in DataFrame
            mask = (
                (df["TrackMate_Track_ID"] == track_id)
                & (df["t"] == spot.frame)
                & (np.abs(df["z"] - spot.z) < 1e-3)
                & (np.abs(df["y"] - spot.y) < 1e-3)
                & (np.abs(df["x"] - spot.x) < 1e-3)
            )
            matching = df[mask]
            if len(matching) == 0:
                continue

            row = matching.iloc[0]
            props = {}

            # Shape features
            if not np.isnan(row.get("Eccentricity_Comp_First", np.nan)):
                props["ECCENTRICITY_COMP_FIRST"] = row["Eccentricity_Comp_First"]
            if not np.isnan(row.get("Eccentricity_Comp_Second", np.nan)):
                props["ECCENTRICITY_COMP_SECOND"] = row["Eccentricity_Comp_Second"]
            if not np.isnan(row.get("Eccentricity_Comp_Third", np.nan)):
                props["ECCENTRICITY_COMP_THIRD"] = row["Eccentricity_Comp_Third"]
            if not np.isnan(row.get("Surface_Area", np.nan)):
                props["SURFACE_AREA"] = row["Surface_Area"]
            if not np.isnan(row.get("Cell_Axis_Z", np.nan)):
                props["CELL_AXIS_Z"] = row["Cell_Axis_Z"]
            if not np.isnan(row.get("Cell_Axis_Y", np.nan)):
                props["CELL_AXIS_Y"] = row["Cell_Axis_Y"]
            if not np.isnan(row.get("Cell_Axis_X", np.nan)):
                props["CELL_AXIS_X"] = row["Cell_Axis_X"]

            # Dynamic features
            props["SPEED"] = row.get("Speed", 0.0)
            props["ACCELERATION"] = row.get("Acceleration", 0.0)
            props["MOTION_ANGLE_Z"] = row.get("Motion_Angle_Z", 0.0)
            props["MOTION_ANGLE_Y"] = row.get("Motion_Angle_Y", 0.0)
            props["MOTION_ANGLE_X"] = row.get("Motion_Angle_X", 0.0)
            props["RADIAL_ANGLE_Z"] = row.get("Radial_Angle_Z", 0.0)
            props["RADIAL_ANGLE_Y"] = row.get("Radial_Angle_Y", 0.0)
            props["RADIAL_ANGLE_X"] = row.get("Radial_Angle_X", 0.0)
            props["DISTANCE_CELL_MASK"] = row.get("Distance_Cell_mask", 0.0)
            props["LOCAL_CELL_DENSITY"] = row.get("Local_Cell_Density", 0)

            if props:
                spot_properties[sid] = props

    return spot_properties


def process_channel(
    channel,
    xml_path,
    seg_image,
    mask_image,
    calibration,
    num_points,
    output_dir,
    save_df,
    normalize_df,
):
    """Process a single channel: compute features, write master XML + CSV."""
    print(f"\n{'='*60}")
    print(f"Processing {channel} channel")
    print(f"XML: {xml_path}")
    print(f"{'='*60}")

    # Compute all features
    tv = TrackVectors(
        xml_path,
        seg_image=seg_image,
        mask=mask_image,
        calibration=calibration,
    )
    df = tv.to_dataframe()

    print(
        f"Computed features for {len(df)} spots, "
        f"{df['TrackMate_Track_ID'].nunique()} tracks, "
        f"{df['Track_ID'].nunique()} tracklets"
    )

    # Build spot properties and write master XML
    spot_props = build_spot_properties(df, tv.xml)
    master_name = (
        f"master_{channel}_{os.path.splitext(os.path.basename(xml_path))[0]}.xml"
    )
    write_trackmate_xml(tv.xml, spot_props, output_dir, output_name=master_name)
    print(f"Master XML written: {os.path.join(output_dir, master_name)}")

    # Save DataFrame (includes tracklet info for all tracks)
    if save_df:
        dataframes_dir = os.path.join(output_dir, "dataframes")
        Path(dataframes_dir).mkdir(parents=True, exist_ok=True)

        raw_csv = os.path.join(dataframes_dir, f"results_dataframe_{channel}.csv")
        df.to_csv(raw_csv, index=False)
        print(f"DataFrame saved: {raw_csv}")

        if normalize_df:
            feature_cols = [c for c in SHAPE_DYNAMIC_FEATURES if c in df.columns]
            df_norm = normalize_features(df, feature_cols)
            norm_csv = os.path.join(
                dataframes_dir, f"results_dataframe_normalized_{channel}.csv"
            )
            df_norm.to_csv(norm_csv, index=False)
            print(f"Normalized DataFrame saved: {norm_csv}")

    return tv


def run_channel_transfer(
    source_xml,
    target_seg_image,
    target_channel_name,
    output_dir,
    target_intensity_image=None,
):
    """Transfer tracks from source XML to target channel segmentation."""
    print(f"\n{'='*60}")
    print(f"Channel transfer: {target_channel_name}")
    print(f"{'='*60}")

    transfer = ChannelTransfer(
        source_xml,
        target_seg_image,
        target_intensity_image=target_intensity_image,
        channel_name=target_channel_name,
    )
    transfer.run()
    transfer.write_xml(output_dir)


@hydra.main(
    version_base="1.3",
    config_path="../conf",
    config_name="scenario_master_tracking",
)
def main(config: MasterTrackingClass):
    # Parameters
    num_points = config.parameters.num_points
    do_nuclei = config.parameters.do_nuclei
    do_membrane = config.parameters.do_membrane
    do_channel_transfer = config.parameters.do_channel_transfer
    transfer_source = config.parameters.transfer_source_channel
    transfer_target = config.parameters.transfer_target_channel
    save_df = config.parameters.save_dataframe
    normalize_df = config.parameters.normalize_dataframe

    # Data paths
    seg_nuclei_dir = config.data_paths.seg_nuclei_directory
    seg_membrane_dir = config.data_paths.seg_membrane_directory
    mask_dir = config.data_paths.mask_directory
    tracking_dir = config.data_paths.tracking_directory
    output_dir = config.data_paths.output_directory
    nuclei_name = config.data_paths.timelapse_nuclei_name
    membrane_name = config.data_paths.timelapse_membrane_name
    calibration_str = config.data_paths.calibration

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Parse calibration
    calibration = None
    if calibration_str:
        calibration = tuple(float(c) for c in calibration_str.split(","))

    # Load mask if specified
    mask_image = None
    if mask_dir:
        mask_path = os.path.join(mask_dir, f"{membrane_name}.tif")
        if os.path.exists(mask_path):
            print(f"Loading mask: {mask_path}")
            mask_image = imread(mask_path)
        else:
            mask_path = os.path.join(mask_dir, f"{nuclei_name}.tif")
            if os.path.exists(mask_path):
                print(f"Loading mask: {mask_path}")
                mask_image = imread(mask_path)

    # Store parsed XMLs for channel transfer
    source_xml = None

    # Process nuclei channel
    if do_nuclei:
        xml_path = os.path.join(tracking_dir, f"nuclei_{nuclei_name}.xml")
        seg_path = os.path.join(seg_nuclei_dir, f"{nuclei_name}.tif")
        print(f"Loading nuclei segmentation: {seg_path}")
        seg_image = imread(seg_path)

        tv = process_channel(
            "nuclei",
            xml_path,
            seg_image,
            mask_image,
            calibration,
            num_points,
            output_dir,
            save_df,
            normalize_df,
        )

        if transfer_source == "nuclei":
            source_xml = tv.xml
        del seg_image

    # Process membrane channel
    if do_membrane:
        xml_path = os.path.join(tracking_dir, f"membrane_{membrane_name}.xml")
        seg_path = os.path.join(seg_membrane_dir, f"{membrane_name}.tif")
        print(f"Loading membrane segmentation: {seg_path}")
        seg_image = imread(seg_path)

        tv = process_channel(
            "membrane",
            xml_path,
            seg_image,
            mask_image,
            calibration,
            num_points,
            output_dir,
            save_df,
            normalize_df,
        )

        if transfer_source == "membrane":
            source_xml = tv.xml
        del seg_image

    # Channel transfer
    if do_channel_transfer and source_xml is not None:
        if transfer_target == "membrane":
            target_seg_path = os.path.join(seg_membrane_dir, f"{membrane_name}.tif")
        else:
            target_seg_path = os.path.join(seg_nuclei_dir, f"{nuclei_name}.tif")

        print(f"Loading target segmentation for transfer: {target_seg_path}")
        target_seg = imread(target_seg_path)

        run_channel_transfer(
            source_xml,
            target_seg,
            transfer_target,
            output_dir,
        )
        del target_seg

    print("\nDone.")


if __name__ == "__main__":
    main()

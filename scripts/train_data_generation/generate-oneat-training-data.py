from pathlib import Path
import os
import glob
import numpy as np
from tifffile import imread

import hydra
from hydra.core.config_store import ConfigStore
from kapoorlabs_lightning.utils import create_event_dataset_h5, normalize_in_chunks
from scenario_generate_oneat import OneatDataClass


configstore = ConfigStore.instance()
configstore.store(name="OneatDataClass", node=OneatDataClass)


@hydra.main(
    config_path="../conf", config_name="scenario_generate_oneat"
)
def main(config: OneatDataClass):

    size_tminus = config.parameters.size_tminus
    size_tplus = config.parameters.size_tplus
    imagex = config.parameters.imagex
    imagey = config.parameters.imagey
    imagez = config.parameters.imagez
    file_type = config.parameters.file_type
    normalizeimage = config.parameters.normalizeimage
    pmin = config.parameters.pmin
    pmax = config.parameters.pmax
    event_name = config.parameters.event_name
    train_split = config.parameters.train_split
    batch_write_size = config.parameters.batch_write_size

    base_data_dir = config.train_data_paths.base_data_dir
    raw_data_dir = os.path.join(base_data_dir, config.train_data_paths.oneat_timelapse_data_raw)
    csv_data_dir = os.path.join(base_data_dir, config.train_data_paths.oneat_timelapse_data_csv)
    seg_data_dir = os.path.join(base_data_dir, config.train_data_paths.oneat_timelapse_data_seg)
    h5_output_path = os.path.join(base_data_dir, config.train_data_paths.oneat_h5_file + '.h5')

    crop_size = (imagex, imagey, imagez, size_tminus, size_tplus)

    raw_files = sorted(glob.glob(os.path.join(raw_data_dir, file_type)))
    seg_files = sorted(glob.glob(os.path.join(seg_data_dir, file_type)))
    csv_files = sorted(glob.glob(os.path.join(csv_data_dir, '*.csv')))

    print(f"Found {len(raw_files)} raw images")
    print(f"Found {len(seg_files)} segmentation images")
    print(f"Found {len(csv_files)} CSV files")

    raw_images = []
    seg_images = []

    for raw_file, seg_file in zip(raw_files, seg_files):
        print(f"Loading {os.path.basename(raw_file)}...")
        raw_img = imread(raw_file)
        seg_img = imread(seg_file)

        if normalizeimage:
            print(f"Normalizing {os.path.basename(raw_file)} in chunks...")
            raw_img = normalize_in_chunks(
                raw_img,
                chunk_steps=50,
                pmin=pmin,
                pmax=pmax,
                dtype=np.float32
            )

        raw_images.append(raw_img)
        seg_images.append(seg_img)

    print(f"Creating H5 dataset at {h5_output_path}...")
    create_event_dataset_h5(
        raw_images=raw_images,
        seg_images=seg_images,
        csv_files=csv_files,
        event_names=event_name,
        h5_output_path=h5_output_path,
        crop_size=crop_size,
        train_split=train_split,
        batch_write_size=batch_write_size,
        raw_files=raw_files
    )

    print(f"H5 dataset created successfully at {h5_output_path}")
    




if __name__ == "__main__":
    main()    
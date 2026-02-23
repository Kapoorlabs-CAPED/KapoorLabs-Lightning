from pathlib import Path
import os
import glob
import numpy as np
from tifffile import imread
import h5py
import pandas as pd

import hydra
from hydra.core.config_store import ConfigStore
from kapoorlabs_lightning.utils import normalize_in_chunks
from scenario_generate_oneat import OneatDataClass


configstore = ConfigStore.instance()
configstore.store(name="OneatDataClass", node=OneatDataClass)


def extract_event_cube(raw_image, seg_image, event, crop_size):
    """Extract a cube around an event."""
    sizex, sizey, sizez, t_minus, t_plus = crop_size
    time = int(event['time'])
    x = int(event['x'])
    y = int(event['y'])
    z = int(event['z'])

    if time < t_minus or time + t_plus + 1 >= raw_image.shape[0]:
        return None

    # Extract temporal window
    t_start = time - t_minus
    t_end = time + t_plus + 1

    # Extract spatial crop
    z_start = max(0, z - sizez // 2)
    z_end = min(raw_image.shape[1], z + sizez // 2)
    y_start = max(0, y - sizey // 2)
    y_end = min(raw_image.shape[2], y + sizey // 2)
    x_start = max(0, x - sizex // 2)
    x_end = min(raw_image.shape[3], x + sizex // 2)

    crop_raw = raw_image[t_start:t_end, z_start:z_end, y_start:y_end, x_start:x_end]
    crop_seg = seg_image[t_start:t_end, z_start:z_end, y_start:y_end, x_start:x_end]

    # Pad if needed
    target_shape = (t_end - t_start, sizez, sizey, sizex)
    if crop_raw.shape != target_shape:
        padded_raw = np.zeros(target_shape, dtype=crop_raw.dtype)
        padded_seg = np.zeros(target_shape, dtype=crop_seg.dtype)

        padded_raw[:crop_raw.shape[0], :crop_raw.shape[1], :crop_raw.shape[2], :crop_raw.shape[3]] = crop_raw
        padded_seg[:crop_seg.shape[0], :crop_seg.shape[1], :crop_seg.shape[2], :crop_seg.shape[3]] = crop_seg

        crop_raw = padded_raw
        crop_seg = padded_seg

    return crop_raw, crop_seg, event['event_label']


def write_batch_to_h5(group, images, segs, labels):
    """Write a batch of data to H5 group."""
    if 'images' not in group:
        group.create_dataset(
            'images',
            data=np.array(images),
            maxshape=(None,) + images[0].shape,
            chunks=True,
            compression='gzip'
        )
        group.create_dataset(
            'segmentations',
            data=np.array(segs),
            maxshape=(None,) + segs[0].shape,
            chunks=True,
            compression='gzip'
        )
        group.create_dataset(
            'labels',
            data=np.array(labels),
            maxshape=(None,),
            chunks=True
        )
    else:
        old_size = group['images'].shape[0]
        new_size = old_size + len(images)
        group['images'].resize(new_size, axis=0)
        group['segmentations'].resize(new_size, axis=0)
        group['labels'].resize(new_size, axis=0)
        group['images'][old_size:new_size] = images
        group['segmentations'][old_size:new_size] = segs
        group['labels'][old_size:new_size] = labels


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
    event_names = config.parameters.event_name
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

    # Create mapping from image basename to files
    image_name_to_idx = {}
    for idx, raw_file in enumerate(raw_files):
        basename = os.path.basename(raw_file)
        image_name = os.path.splitext(basename)[0]
        image_name_to_idx[image_name] = idx

    # Collect all event metadata (lightweight)
    print("Collecting event metadata...")
    all_events = []
    for csv_file in csv_files:
        csv_name = os.path.basename(csv_file)

        # Find which event and image this CSV corresponds to
        for event_idx, event_name in enumerate(event_names):
            if event_name in csv_name:
                prefix = f"oneat_{event_name}_"
                if csv_name.startswith(prefix):
                    image_name = csv_name[len(prefix):].replace('.csv', '')

                    if image_name in image_name_to_idx:
                        raw_idx = image_name_to_idx[image_name]
                        clicks_df = pd.read_csv(csv_file)
                        # Normalize column names
                        clicks_df.columns = [col.lower() for col in clicks_df.columns]

                        for _, row in clicks_df.iterrows():
                            all_events.append({
                                'raw_idx': raw_idx,
                                'event_label': event_idx,
                                'event_name': event_name,
                                'time': row.get('t', row.get('time', 0)),
                                'x': row.get('x', 0),
                                'y': row.get('y', 0),
                                'z': row.get('z', 0)
                            })
                    break

    print(f"Found {len(all_events)} total events")

    # Shuffle and split
    np.random.shuffle(all_events)
    train_size = int(len(all_events) * train_split)
    train_events = all_events[:train_size]
    val_events = all_events[train_size:]

    print(f"Train events: {len(train_events)}")
    print(f"Val events: {len(val_events)}")

    # Create H5 file
    print(f"\nCreating H5 dataset at {h5_output_path}...")
    with h5py.File(h5_output_path, 'w') as h5f:
        train_grp = h5f.create_group('train')
        val_grp = h5f.create_group('val')

        # Process train and val separately
        for split_name, split_events, split_grp in [
            ('train', train_events, train_grp),
            ('val', val_events, val_grp)
        ]:
            print(f"\nProcessing {split_name} split...")

            # Group events by image to process one image at a time
            events_by_image = {}
            for event in split_events:
                img_idx = event['raw_idx']
                if img_idx not in events_by_image:
                    events_by_image[img_idx] = []
                events_by_image[img_idx].append(event)

            # Process each image
            batch_images = []
            batch_segs = []
            batch_labels = []
            total_processed = 0

            for img_idx in sorted(events_by_image.keys()):
                raw_file = raw_files[img_idx]
                seg_file = seg_files[img_idx]

                print(f"  Loading {os.path.basename(raw_file)}...")

                # Load ONE image at a time
                raw_img = imread(raw_file)
                seg_img = imread(seg_file)

                if normalizeimage:
                    raw_img = normalize_in_chunks(
                        raw_img,
                        chunk_steps=50,
                        pmin=pmin,
                        pmax=pmax,
                        dtype=np.float32
                    )

                # Extract all events from this image
                for event in events_by_image[img_idx]:
                    result = extract_event_cube(raw_img, seg_img, event, crop_size)
                    if result is not None:
                        crop_image, crop_seg, label = result
                        batch_images.append(crop_image)
                        batch_segs.append(crop_seg)
                        batch_labels.append(label)

                        # Flush batch when it reaches size
                        if len(batch_images) >= batch_write_size:
                            write_batch_to_h5(split_grp, batch_images, batch_segs, batch_labels)
                            total_processed += len(batch_images)
                            print(f"    Flushed {len(batch_images)} samples (total: {total_processed})")
                            batch_images = []
                            batch_segs = []
                            batch_labels = []

                # Delete image from memory immediately
                del raw_img
                del seg_img

            # Flush remaining batch
            if len(batch_images) > 0:
                write_batch_to_h5(split_grp, batch_images, batch_segs, batch_labels)
                total_processed += len(batch_images)
                print(f"    Flushed final {len(batch_images)} samples (total: {total_processed})")

    print(f"\nH5 dataset created successfully at {h5_output_path}")


if __name__ == "__main__":
    main()

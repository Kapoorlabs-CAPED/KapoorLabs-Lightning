import torch
import numpy as np
from torch.utils.data import Dataset
from tifffile import imread
from .utils import normalize_in_chunks


class OneatPredictionDataset(Dataset):
    """
    Dataset for ONEAT prediction that yields temporal windows for each timepoint.
    """

    def __init__(
        self,
        raw_file,
        seg_file,
        size_tminus=1,
        size_tplus=1,
        normalize=True,
        pmin=1.0,
        pmax=99.8,
        chunk_steps=50,
    ):
        self.raw_file = raw_file
        self.seg_file = seg_file
        self.size_tminus = size_tminus
        self.size_tplus = size_tplus
        self.imaget = size_tminus + size_tplus + 1
        self.normalize = normalize
        self.pmin = pmin
        self.pmax = pmax
        self.chunk_steps = chunk_steps

        # Load images
        print(f"Loading raw image: {raw_file}")
        self.raw_image = imread(raw_file)  # Shape: (T, Z, Y, X)

        print(f"Loading seg image: {seg_file}")
        self.seg_image = imread(seg_file)  # Shape: (T, Z, Y, X)

        # Normalize raw image in chunks
        if self.normalize:
            print("Normalizing image in chunks...")
            self.raw_image = normalize_in_chunks(
                self.raw_image,
                chunk_steps=self.chunk_steps,
                pmin=self.pmin,
                pmax=self.pmax,
                dtype=np.float32
            )

        self.num_timepoints = self.raw_image.shape[0]

        # Valid timepoints (accounting for temporal window)
        self.valid_timepoints = list(range(self.size_tminus, self.num_timepoints - self.size_tplus))

        print(f"Dataset created with {len(self.valid_timepoints)} valid timepoints")

    def __len__(self):
        return len(self.valid_timepoints)

    def __getitem__(self, idx):
        t = self.valid_timepoints[idx]

        # Extract temporal window
        start_t = t - self.size_tminus
        end_t = t + self.size_tplus + 1

        temporal_raw = self.raw_image[start_t:end_t]  # Shape: (imaget, Z, Y, X)
        temporal_seg = self.seg_image[start_t:end_t]  # Shape: (imaget, Z, Y, X)

        # Convert to tensors
        temporal_raw = torch.from_numpy(temporal_raw).float()
        temporal_seg = torch.from_numpy(temporal_seg).long()

        # Metadata
        metadata = {
            'filename': self.raw_file,
            'timepoint': t
        }

        return temporal_raw, temporal_seg, torch.tensor(t), metadata

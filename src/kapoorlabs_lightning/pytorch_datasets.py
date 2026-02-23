import json
import os
from glob import glob
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
from pyntcloud import PyntCloud
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn import preprocessing
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import networkx as nx
from skimage import measure
from tifffile import imread
from lightning import LightningDataModule
from typing import Any, Dict, Optional



class MitosisDataset(Dataset):
    def __init__(self, arrays, labels):
        self.arrays = arrays
        self.labels = labels
        self.input_channels = arrays.shape[2]

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        array = self.arrays[idx]
        array = torch.tensor(array).permute(1, 0).float()
        label = torch.tensor(self.labels[idx])

        return array, label


class H5MitosisDataset(Dataset):
    def __init__(self, h5_file, data_key, label_key, num_classes = 3, transforms = None):
        self.h5_file = h5_file
        self.data_key = data_key
        self.label_key = label_key
        self.transforms = transforms
        self.data_label = h5py.File(self.h5_file, "r")
        self.data = self.data_label[data_key]
        self.targets = self.data_label[label_key]
        self.input_channels = np.asarray(self.data[0]).shape[1]
        self.class_weights_dict = self._compute_class_weights(num_classes)
        print(f'Class weights computed {self.class_weights_dict}')
  
    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx):
       
            array = torch.from_numpy(np.asarray(self.data[idx])).permute(1, 0).float()
      
            label = torch.from_numpy(np.asarray(self.targets[idx]))

            if self.transforms:
                array = self.transforms(array)

            return array, label
    
    def _compute_class_weights(self, num_classes):
        """
        Compute class weights directly using self.targets.

        Args:
            num_classes (int): Number of classes.

        Returns:
            np.ndarray: Array of class weights.
        """
        class_counts = np.bincount(self.targets[:], minlength=num_classes)

        total_samples = len(self.targets)

        class_weights = total_samples / (num_classes * np.maximum(class_counts, 1))  

        class_weights_dict = {class_idx: weight for class_idx, weight in enumerate(class_weights)}

        return class_weights_dict

    
class H5VisionDataset(Dataset):
    def __init__(
        self,
        h5_file: str,
        split: str = "train",
        transforms=None,
        return_segmentation: bool = False,
        num_classes: int = None,
        compute_class_weights: bool = False,
    ):
        self.h5_file = h5_file
        self.split = split
        self.transforms = transforms
        self.return_segmentation = return_segmentation
        self.num_classes = num_classes

        self.h5_handle = h5py.File(self.h5_file, "r", swmr=True)

        if split not in self.h5_handle:
            raise ValueError(f"Split '{split}' not found in H5 file. Available: {list(self.h5_handle.keys())}")

        self.group = self.h5_handle[split]

        if "images" not in self.group or "labels" not in self.group:
            raise ValueError(f"Split '{split}' must contain 'images' and 'labels' datasets")

        self.images_dataset = self.group["images"]
        self.labels_dataset = self.group["labels"]

        if return_segmentation and "segmentations" in self.group:
            self.segmentations_dataset = self.group["segmentations"]
        else:
            self.segmentations_dataset = None

        self.length = len(self.labels_dataset)

        self.class_weights_dict = None
        if compute_class_weights and num_classes is not None:
            self.class_weights_dict = self._compute_class_weights()
            print(f"Class weights computed: {self.class_weights_dict}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images_dataset[idx]).float()
        label = torch.from_numpy(np.asarray(self.labels_dataset[idx])).float()

        if self.transforms is not None:
            image = self.transforms(image)

        if self.return_segmentation and self.segmentations_dataset is not None:
            segmentation = torch.from_numpy(self.segmentations_dataset[idx]).long()
            return image, label, segmentation

        return image, label

    def _compute_class_weights(self):
        all_labels = self.labels_dataset[:]

        # Handle YOLO labels: [x,y,z,t,h,w,d,c] + [one-hot categories]
        # box_vector = 8, so categories start at index 8
        if all_labels.ndim > 1:
            box_vector_len = 8
            one_hot_categories = all_labels[:, box_vector_len:]
            class_indices = np.argmax(one_hot_categories, axis=1)
        else:
            class_indices = all_labels

        class_counts = np.bincount(class_indices.astype(int), minlength=self.num_classes)
        total_samples = len(class_indices)
        class_weights = total_samples / (self.num_classes * np.maximum(class_counts, 1))
        class_weights_dict = {class_idx: weight for class_idx, weight in enumerate(class_weights)}
        return class_weights_dict

    def get_class_weights(self):
        if self.class_weights_dict is None:
            raise ValueError("Class weights not computed. Set compute_class_weights=True in constructor.")
        return self.class_weights_dict

    def get_class_weights_tensor(self):
        if self.class_weights_dict is None:
            raise ValueError("Class weights not computed. Set compute_class_weights=True in constructor.")
        weights = [self.class_weights_dict[i] for i in range(self.num_classes)]
        return torch.FloatTensor(weights)

    def __del__(self):
        if hasattr(self, 'h5_handle'):
            self.h5_handle.close()    


class PointCloudDataset(Dataset):
    def __init__(self, points_dir, centre=True, scale_z=1.0, scale_xy=1.0):
        self.points_dir = points_dir
        self.centre = centre
        self.scale_z = scale_z
        self.scale_xy = scale_xy
        self.p = Path(self.points_dir)
        self.files = list(self.p.glob("**/*.ply"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # read the image
        file = self.files[idx]
        point_cloud = PyntCloud.from_file(str(file))
        mean = 0
        point_cloud = torch.tensor(point_cloud.points.values)
        if self.centre:
            mean = torch.mean(point_cloud, 0)

        scale = torch.tensor([[self.scale_z, self.scale_xy, self.scale_xy]])
        point_cloud = (point_cloud - mean) / scale

        return point_cloud


class GenericDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self._dataset = dataset
        self.transform = transform

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    def __call__(self):
        return self._dataset


class GenericDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_train: Dataset = None,
        dataset_test: Dataset = None,
        dataset_val: Dataset = None,
        batch_size_train: int = 64,
        batch_size_val: int = 64,
        batch_size_test: int = 64,
        num_workers_train: int = 1,
        num_workers_val: int = 1,
        num_workers_test: int = 1,
        prefetch_factor_train: int = 4,
        prefetch_factor_val: int = 4,
        prefetch_factor_test: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        sampler=None,
    ):
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["dataset_train", "dataset_val", "dataset_test"],
        )
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.num_workers_train = num_workers_train
        self.num_workers_val = num_workers_val
        self.num_workers_test = num_workers_test
        self.prefetch_factor_train = prefetch_factor_train
        self.prefetch_factor_val = prefetch_factor_val
        self.prefetch_factor_test = prefetch_factor_test
        self.pin_memory = pin_memory
        self.sampler = sampler
        self.persistent_workers = persistent_workers

        if self.num_workers_train == 0:
            self.prefetch_factor_train = None
            self.persistent_workers = None
        if self.num_workers_val == 0:
            self.prefetch_factor_val = None
            self.persistent_workers = None
        if self.num_workers_test == 0:
            self.prefetch_factor_test = None
            self.persistent_workers = None

        self.generic_dataset_train = GenericDataset(self.dataset_train)
        self.generic_dataset_val = GenericDataset(self.dataset_val)
        self.generic_dataset_test = GenericDataset(self.dataset_test)

        self.setup()

    def setup(self, stage: str = "train"):
        self.data_train = self.generic_dataset_train()
        self.data_val = self.generic_dataset_val()
        self.data_test = self.generic_dataset_test()

    def train_dataloader(self):
        if self.sampler is None:
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size_train,
                shuffle=True,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers_train,
                prefetch_factor=self.prefetch_factor_train,
                persistent_workers=self.persistent_workers,
            )
        else:
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size_train,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers_train,
                prefetch_factor=self.prefetch_factor_train,
                persistent_workers=self.persistent_workers,
                sampler=self.sampler,
            )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_val,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers_val,
            prefetch_factor=self.prefetch_factor_val,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_test,
            shuffle=False,
            num_workers=self.num_workers_test,
            prefetch_factor=self.prefetch_factor_test,
            persistent_workers=self.persistent_workers,
        )

    def teardown(self, stage: Optional[str] = None):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass

from typing import List
import torch
from torchvision import transforms
import numpy as np

class Transforms:
    def __init__(
        self,
        aug_str: str,
        mean: List = [0.4987, 0.4702, 0.4050],
        std: List = [0.2711, 0.2635, 0.2810],
    ):
        """
        Initializes the Transforms class.

        Args:
            aug_str (str): String containing the desired augmentations.
            mean (List[float], optional): List of means for each color channel, used for normalization.
            std (List[float], optional): List of standard deviations for each color channel, used for normalization.
        """
        self.aug_str = aug_str
        self.mean = mean
        self.std = std
        self.compose_transform()

    def compose_transform(self):
        """
        Composes the transforms listed in `aug_str` and stores them in `self.transform`.
        """
        transform_list = []
        if "trivialaug" in self.aug_str:
            transform_list.append(transforms.TrivialAugmentWide())

        if "randaug" in self.aug_str:
            transform_list.append(transforms.RandAugment())

        if "normalize" in self.aug_str:
            transform_list.append(transforms.ConvertImageDtype(torch.float))
            transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))
        if "resnet" in self.aug_str:
            transform_list.append(transforms.Resize((224, 224)))
            transform_list.append(transforms.ToTensor())
            transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))

        self.transform = transforms.Compose(transform_list)

    def get_transform(self):
        """
        Returns the composed transform.
        """
        return self.transform



class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=0.01):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise

class RandomTimeShift(torch.nn.Module):
    def __init__(self, max_shift=2):
        super().__init__()
        self.max_shift = max_shift

    def forward(self, x):
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        return torch.roll(x, shifts=shift, dims=-1)

class RandomScaling(torch.nn.Module):
    def __init__(self, min_scale=0.9, max_scale=1.1):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, x):
        scale = np.random.uniform(self.min_scale, self.max_scale)
        return x * scale

class RandomMasking(torch.nn.Module):
    def __init__(self, max_mask_ratio=0.2):
        super().__init__()
        self.max_mask_ratio = max_mask_ratio

    def forward(self, x):
        mask_ratio = np.random.uniform(0, self.max_mask_ratio)
        num_mask = int(mask_ratio * x.shape[-1])
        mask_indices = np.random.choice(x.shape[-1], num_mask, replace=False)
        x[:, mask_indices] = 0  
        return x


def get_transforms(mean = 0.0, std=0.02, min_scale= 0.95,max_shift=1.05,  max_scale=1.05, max_mask_ratio=0.1):
    time_series_transforms = transforms.Compose([
            AddGaussianNoise(mean=mean, std=std),
            RandomTimeShift(max_shift=max_shift),
            RandomScaling(min_scale=min_scale, max_scale=max_scale),
            RandomMasking(max_mask_ratio=max_mask_ratio),
        ])
    
    return time_series_transforms



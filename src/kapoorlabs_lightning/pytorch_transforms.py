from typing import List
import torch
from torchvision import transforms


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

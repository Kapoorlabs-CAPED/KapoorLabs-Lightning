import torch
from torchvision import transforms as T
from .time_series_transforms import (
    AddGaussianNoise,
    RandomScaling,
    RandomMasking,
)


class CellFateTrainPresetLight:
    def __init__(
        self,
        gaussian_noise_std: float = 0.01,
        gaussian_noise_p: float = 0.3,
        min_scale: float = 0.98,
        max_scale: float = 1.02,
    ):
        transform_list = [
            AddGaussianNoise(mean=0.0, std=gaussian_noise_std),
            RandomScaling(min_scale=min_scale, max_scale=max_scale),
        ]
        self.transforms = T.Compose(transform_list)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transforms(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class CellFateTrainPresetMedium:
    def __init__(
        self,
        gaussian_noise_std: float = 0.02,
        gaussian_noise_p: float = 0.5,
        min_scale: float = 0.95,
        max_scale: float = 1.05,
        max_mask_ratio: float = 0.1,
    ):
        transform_list = [
            AddGaussianNoise(mean=0.0, std=gaussian_noise_std),
            RandomScaling(min_scale=min_scale, max_scale=max_scale),
            RandomMasking(max_mask_ratio=max_mask_ratio),
        ]
        self.transforms = T.Compose(transform_list)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transforms(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class CellFateTrainPresetHeavy:
    def __init__(
        self,
        gaussian_noise_std: float = 0.03,
        gaussian_noise_p: float = 0.6,
        min_scale: float = 0.9,
        max_scale: float = 1.1,
        max_mask_ratio: float = 0.2,
    ):
        transform_list = [
            AddGaussianNoise(mean=0.0, std=gaussian_noise_std),
            RandomScaling(min_scale=min_scale, max_scale=max_scale),
            RandomMasking(max_mask_ratio=max_mask_ratio),
        ]
        self.transforms = T.Compose(transform_list)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transforms(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


__all__ = [
    "CellFateTrainPresetLight",
    "CellFateTrainPresetMedium",
    "CellFateTrainPresetHeavy",
]

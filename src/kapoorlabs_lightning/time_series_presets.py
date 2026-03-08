import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms as T


class AddGaussianNoise(nn.Module):
    def __init__(self, mean: float = 0.0, std: float = 0.01):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class RandomScaling(nn.Module):
    def __init__(self, min_scale: float = 0.9, max_scale: float = 1.1):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = np.random.uniform(self.min_scale, self.max_scale)
        return x * scale

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_scale={self.min_scale}, max_scale={self.max_scale})"


class RandomMasking(nn.Module):
    def __init__(self, max_mask_ratio: float = 0.2):
        super().__init__()
        self.max_mask_ratio = max_mask_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask_ratio = np.random.uniform(0, self.max_mask_ratio)
        num_mask = int(mask_ratio * x.shape[-1])
        if num_mask > 0:
            mask_indices = np.random.choice(x.shape[-1], num_mask, replace=False)
            x[:, mask_indices] = 0
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_mask_ratio={self.max_mask_ratio})"


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
    "AddGaussianNoise",
    "RandomScaling",
    "RandomMasking",
    "CellFateTrainPresetLight",
    "CellFateTrainPresetMedium",
    "CellFateTrainPresetHeavy",
]

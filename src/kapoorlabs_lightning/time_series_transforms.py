import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms


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


class RandomTimeShift(nn.Module):
    def __init__(self, max_shift: int = 2):
        super().__init__()
        self.max_shift = max_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        return torch.roll(x, shifts=shift, dims=-1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_shift={self.max_shift})"


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


class RandomTimePermutation(nn.Module):
    def __init__(self, segment_size: int = 3, p: float = 0.5):
        super().__init__()
        self.segment_size = segment_size
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() < self.p:
            time_steps = x.shape[-1]
            num_segments = time_steps // self.segment_size
            if num_segments > 1:
                segments = x[..., :num_segments * self.segment_size].reshape(
                    *x.shape[:-1], num_segments, self.segment_size
                )
                perm_idx = torch.randperm(num_segments)
                segments = segments[..., perm_idx, :]
                x[..., :num_segments * self.segment_size] = segments.reshape(
                    *x.shape[:-1], num_segments * self.segment_size
                )
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(segment_size={self.segment_size}, p={self.p})"


class RandomTimeWarping(nn.Module):
    def __init__(self, sigma: float = 0.2, p: float = 0.5):
        super().__init__()
        self.sigma = sigma
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() < self.p:
            time_steps = x.shape[-1]
            warp = np.cumsum(np.random.randn(time_steps) * self.sigma)
            warp = (warp - warp.min()) / (warp.max() - warp.min()) * (time_steps - 1)
            indices = torch.from_numpy(warp).long().clamp(0, time_steps - 1)
            x = x[..., indices]
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sigma={self.sigma}, p={self.p})"


def get_time_series_transforms(
    mean: float = 0.0,
    std: float = 0.02,
    min_scale: float = 0.95,
    max_shift: int = 1,
    max_scale: float = 1.05,
    max_mask_ratio: float = 0.1,
):
    time_series_transforms = transforms.Compose([
        AddGaussianNoise(mean=mean, std=std),
        RandomTimeShift(max_shift=max_shift),
        RandomScaling(min_scale=min_scale, max_scale=max_scale),
        RandomMasking(max_mask_ratio=max_mask_ratio),
    ])

    return time_series_transforms


__all__ = [
    "AddGaussianNoise",
    "RandomTimeShift",
    "RandomScaling",
    "RandomMasking",
    "RandomTimePermutation",
    "RandomTimeWarping",
    "get_time_series_transforms",
]

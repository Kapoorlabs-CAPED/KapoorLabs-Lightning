"""
Paired transforms for CARE (Content-Aware image REstoration).

Geometric transforms (flip, rotation) are applied identically to both
low SNR input and high SNR target. Intensity augmentations are applied
only to the input.
"""

import torch
import torch.nn as nn
from .oneat_transforms import PercentileNormalize, ToFloat32


class PairedRandomSpatialFlip(nn.Module):
    """Apply identical random flips to a pair of 3D volumes (ZYX)."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, low: torch.Tensor, high: torch.Tensor):
        if low.dim() == 3:
            # Z flip
            if torch.rand(1).item() < self.p:
                low = torch.flip(low, dims=[0])
                high = torch.flip(high, dims=[0])
            # Y flip
            if torch.rand(1).item() < self.p:
                low = torch.flip(low, dims=[1])
                high = torch.flip(high, dims=[1])
            # X flip
            if torch.rand(1).item() < self.p:
                low = torch.flip(low, dims=[2])
                high = torch.flip(high, dims=[2])
        return low, high


class PairedRandomRotation90(nn.Module):
    """Apply identical random 90-degree rotation in YX plane to a pair."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, low: torch.Tensor, high: torch.Tensor):
        if torch.rand(1).item() < self.p:
            k = torch.randint(0, 4, (1,)).item()
            if low.dim() == 3:
                low = torch.rot90(low, k, dims=[1, 2])
                high = torch.rot90(high, k, dims=[1, 2])
        return low, high


class InputGaussianNoise(nn.Module):
    """Add Gaussian noise to input only (not target)."""

    def __init__(self, std: float = 0.01, p: float = 0.5):
        super().__init__()
        self.std = std
        self.p = p

    def forward(self, low: torch.Tensor, high: torch.Tensor):
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(low) * self.std
            low = low + noise
        return low, high


class PairedPercentileNormalize(nn.Module):
    """Apply percentile normalization independently to each volume."""

    def __init__(self, pmin: float = 0.1, pmax: float = 99.9, eps: float = 1e-8):
        super().__init__()
        self.norm = PercentileNormalize(pmin=pmin, pmax=pmax, eps=eps)

    def forward(self, low: torch.Tensor, high: torch.Tensor):
        return self.norm(low), self.norm(high)


class PairedToFloat32(nn.Module):
    """Convert both volumes to float32."""

    def forward(self, low: torch.Tensor, high: torch.Tensor):
        return low.to(torch.float32), high.to(torch.float32)


__all__ = [
    "PairedRandomSpatialFlip",
    "PairedRandomRotation90",
    "InputGaussianNoise",
    "PairedPercentileNormalize",
    "PairedToFloat32",
]

"""
CARE transform presets for paired low/high SNR training data.

Light: percentile norm + random flip
Medium: + rotation90 + light Gaussian noise on input
Heavy: + stronger noise + intensity scaling on input
Eval: percentile norm only
"""

import torch.nn as nn

from .care_transforms import (
    PairedToFloat32,
    PairedPercentileNormalize,
    PairedRandomSpatialFlip,
    PairedRandomRotation90,
    InputGaussianNoise,
)


class CareTrainPresetLight(nn.Module):
    def __init__(self, pmin=0.1, pmax=99.9, spatial_flip_p=0.5):
        super().__init__()
        self.to_float = PairedToFloat32()
        self.normalize = PairedPercentileNormalize(pmin=pmin, pmax=pmax)
        self.flip = PairedRandomSpatialFlip(p=spatial_flip_p)

    def forward(self, low, high):
        low, high = self.to_float(low, high)
        low, high = self.normalize(low, high)
        low, high = self.flip(low, high)
        return low, high


class CareTrainPresetMedium(nn.Module):
    def __init__(
        self,
        pmin=0.1,
        pmax=99.9,
        spatial_flip_p=0.5,
        rotation_p=0.5,
        gaussian_noise_std=0.01,
        gaussian_noise_p=0.3,
    ):
        super().__init__()
        self.to_float = PairedToFloat32()
        self.normalize = PairedPercentileNormalize(pmin=pmin, pmax=pmax)
        self.flip = PairedRandomSpatialFlip(p=spatial_flip_p)
        self.rotate = PairedRandomRotation90(p=rotation_p)
        self.noise = InputGaussianNoise(std=gaussian_noise_std, p=gaussian_noise_p)

    def forward(self, low, high):
        low, high = self.to_float(low, high)
        low, high = self.normalize(low, high)
        low, high = self.flip(low, high)
        low, high = self.rotate(low, high)
        low, high = self.noise(low, high)
        return low, high


class CareTrainPresetHeavy(nn.Module):
    def __init__(
        self,
        pmin=0.1,
        pmax=99.9,
        spatial_flip_p=0.5,
        rotation_p=0.5,
        gaussian_noise_std=0.03,
        gaussian_noise_p=0.5,
    ):
        super().__init__()
        self.to_float = PairedToFloat32()
        self.normalize = PairedPercentileNormalize(pmin=pmin, pmax=pmax)
        self.flip = PairedRandomSpatialFlip(p=spatial_flip_p)
        self.rotate = PairedRandomRotation90(p=rotation_p)
        self.noise = InputGaussianNoise(std=gaussian_noise_std, p=gaussian_noise_p)

    def forward(self, low, high):
        low, high = self.to_float(low, high)
        low, high = self.normalize(low, high)
        low, high = self.flip(low, high)
        low, high = self.rotate(low, high)
        low, high = self.noise(low, high)
        return low, high


class CareEvalPreset(nn.Module):
    def __init__(self, pmin=0.1, pmax=99.9):
        super().__init__()
        self.to_float = PairedToFloat32()
        self.normalize = PairedPercentileNormalize(pmin=pmin, pmax=pmax)

    def forward(self, low, high):
        low, high = self.to_float(low, high)
        low, high = self.normalize(low, high)
        return low, high


__all__ = [
    "CareTrainPresetLight",
    "CareTrainPresetMedium",
    "CareTrainPresetHeavy",
    "CareEvalPreset",
]

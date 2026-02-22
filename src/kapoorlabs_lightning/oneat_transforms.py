import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class AddGaussianNoise(nn.Module):
    def __init__(self, mean: float = 0.0, std: float = 0.01, p: float = 0.5):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(x) * self.std + self.mean
            return x + noise
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, p={self.p})"


class AddPoissonNoise(nn.Module):
    def __init__(self, scale: float = 1.0, p: float = 0.5):
        super().__init__()
        self.scale = scale
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            x_scaled = x * self.scale
            x_scaled = torch.clamp(x_scaled, min=0)
            noise = torch.poisson(x_scaled) / self.scale
            return noise
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale={self.scale}, p={self.p})"


class SpatialGaussianBlur(nn.Module):
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0, p: float = 0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            if x.dim() == 4:
                t, z, y, x_dim = x.shape
                x_blurred = torch.zeros_like(x)
                for i in range(t):
                    for j in range(z):
                        slice_2d = x[i, j].unsqueeze(0).unsqueeze(0)
                        blurred = F.gaussian_blur(slice_2d, [self.kernel_size, self.kernel_size], [self.sigma, self.sigma])
                        x_blurred[i, j] = blurred.squeeze()
                return x_blurred
            elif x.dim() == 5:
                b, t, z, y, x_dim = x.shape
                x_blurred = torch.zeros_like(x)
                for batch in range(b):
                    for i in range(t):
                        for j in range(z):
                            slice_2d = x[batch, i, j].unsqueeze(0).unsqueeze(0)
                            blurred = F.gaussian_blur(slice_2d, [self.kernel_size, self.kernel_size], [self.sigma, self.sigma])
                            x_blurred[batch, i, j] = blurred.squeeze()
                return x_blurred
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, sigma={self.sigma}, p={self.p})"


class RandomBrightnessContrast(nn.Module):
    def __init__(self, brightness_range: Tuple[float, float] = (0.9, 1.1),
                 contrast_range: Tuple[float, float] = (0.9, 1.1), p: float = 0.5):
        super().__init__()
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            brightness = torch.FloatTensor(1).uniform_(*self.brightness_range).item()
            contrast = torch.FloatTensor(1).uniform_(*self.contrast_range).item()

            mean = x.mean()
            x = (x - mean) * contrast + mean
            x = x * brightness

            return torch.clamp(x, 0, 1)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(brightness_range={self.brightness_range}, contrast_range={self.contrast_range}, p={self.p})"


class RandomSpatialFlip(nn.Module):
    def __init__(self, flip_xy: bool = True, flip_xz: bool = True, flip_yz: bool = True, p: float = 0.5):
        super().__init__()
        self.flip_xy = flip_xy
        self.flip_xz = flip_xz
        self.flip_yz = flip_yz
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            t, z, y, x_dim = x.shape

            if self.flip_xy and torch.rand(1).item() < self.p:
                x = torch.flip(x, dims=[2])

            if self.flip_xz and torch.rand(1).item() < self.p:
                x = torch.flip(x, dims=[1])

            if self.flip_yz and torch.rand(1).item() < self.p:
                x = torch.flip(x, dims=[3])

        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(flip_xy={self.flip_xy}, flip_xz={self.flip_xz}, flip_yz={self.flip_yz}, p={self.p})"


class RandomSpatialRotation90(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            k = torch.randint(0, 4, (1,)).item()
            if x.dim() == 4:
                x = torch.rot90(x, k, dims=[2, 3])
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomIntensityScaling(nn.Module):
    def __init__(self, scale_range: Tuple[float, float] = (0.9, 1.1), p: float = 0.5):
        super().__init__()
        self.scale_range = scale_range
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            scale = torch.FloatTensor(1).uniform_(*self.scale_range).item()
            return torch.clamp(x * scale, 0, 1)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale_range={self.scale_range}, p={self.p})"


class PercentileNormalize(nn.Module):
    def __init__(self, pmin: float = 1.0, pmax: float = 99.8, eps: float = 1e-8):
        super().__init__()
        self.pmin = pmin
        self.pmax = pmax
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mi = torch.quantile(x.flatten(), self.pmin / 100.0)
        ma = torch.quantile(x.flatten(), self.pmax / 100.0)
        x = (x - mi) / (ma - mi + self.eps)
        return torch.clamp(x, 0, 1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pmin={self.pmin}, pmax={self.pmax})"


class MinMaxNormalize(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mi = x.min()
        ma = x.max()
        return (x - mi) / (ma - mi + self.eps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ToFloat32(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ElasticDeformation(nn.Module):
    def __init__(self, alpha: float = 10.0, sigma: float = 3.0, p: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            if x.dim() == 4:
                t, z, y, x_dim = x.shape

                dx = torch.randn(y, x_dim) * self.alpha
                dy = torch.randn(y, x_dim) * self.alpha

                dx = F.gaussian_blur(dx.unsqueeze(0).unsqueeze(0), kernel_size=int(self.sigma * 4) | 1, sigma=self.sigma).squeeze()
                dy = F.gaussian_blur(dy.unsqueeze(0).unsqueeze(0), kernel_size=int(self.sigma * 4) | 1, sigma=self.sigma).squeeze()

                grid_y, grid_x = torch.meshgrid(torch.arange(y), torch.arange(x_dim), indexing='ij')
                grid_y = grid_y.float() + dy
                grid_x = grid_x.float() + dx

                grid_y = 2.0 * grid_y / (y - 1) - 1.0
                grid_x = 2.0 * grid_x / (x_dim - 1) - 1.0

                grid = torch.stack([grid_x, grid_y], dim=-1)

                x_deformed = torch.zeros_like(x)
                for i in range(t):
                    for j in range(z):
                        slice_2d = x[i, j].unsqueeze(0).unsqueeze(0)
                        deformed = F.grid_sample(slice_2d, grid.unsqueeze(0), mode='bilinear', padding_mode='border', align_corners=True)
                        x_deformed[i, j] = deformed.squeeze()

                return x_deformed
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha}, sigma={self.sigma}, p={self.p})"


__all__ = [
    "AddGaussianNoise",
    "AddPoissonNoise",
    "SpatialGaussianBlur",
    "RandomBrightnessContrast",
    "RandomSpatialFlip",
    "RandomSpatialRotation90",
    "RandomIntensityScaling",
    "PercentileNormalize",
    "MinMaxNormalize",
    "ToFloat32",
    "ElasticDeformation",
]

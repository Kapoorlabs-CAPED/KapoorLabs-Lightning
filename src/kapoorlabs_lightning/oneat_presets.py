import torch
from torchvision import transforms as T
from .oneat_transforms import (
    AddGaussianNoise,
    AddPoissonNoise,
    SpatialGaussianBlur,
    RandomBrightnessContrast,
    RandomSpatialFlip,
    RandomSpatialRotation90,
    RandomIntensityScaling,
    PercentileNormalize,
    MinMaxNormalize,
    ToFloat32,
    ElasticDeformation,
)


class OneatTrainPresetLight:
    def __init__(
        self,
        gaussian_noise_std: float = 0.01,
        gaussian_noise_p: float = 0.3,
        brightness_range: tuple = (0.95, 1.05),
        contrast_range: tuple = (0.95, 1.05),
        brightness_contrast_p: float = 0.5,
        spatial_flip_p: float = 0.5,
        percentile_norm: bool = True,
        pmin: float = 1.0,
        pmax: float = 99.8,
    ):
        transform_list = [ToFloat32()]

        if percentile_norm:
            transform_list.append(PercentileNormalize(pmin=pmin, pmax=pmax))
        else:
            transform_list.append(MinMaxNormalize())

        if gaussian_noise_std > 0:
            transform_list.append(
                AddGaussianNoise(mean=0.0, std=gaussian_noise_std, p=gaussian_noise_p)
            )

        if brightness_range and contrast_range:
            transform_list.append(
                RandomBrightnessContrast(
                    brightness_range=brightness_range,
                    contrast_range=contrast_range,
                    p=brightness_contrast_p,
                )
            )

        if spatial_flip_p > 0:
            transform_list.append(
                RandomSpatialFlip(
                    flip_xy=True, flip_xz=False, flip_yz=False, p=spatial_flip_p
                )
            )

        self.transforms = T.Compose(transform_list)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transforms(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class OneatTrainPresetMedium:
    def __init__(
        self,
        gaussian_noise_std: float = 0.02,
        gaussian_noise_p: float = 0.5,
        poisson_noise_scale: float = 1.0,
        poisson_noise_p: float = 0.3,
        blur_kernel_size: int = 3,
        blur_sigma: float = 1.0,
        blur_p: float = 0.3,
        brightness_range: tuple = (0.9, 1.1),
        contrast_range: tuple = (0.9, 1.1),
        brightness_contrast_p: float = 0.5,
        intensity_scale_range: tuple = (0.95, 1.05),
        intensity_scale_p: float = 0.3,
        spatial_flip_p: float = 0.5,
        rotation_p: float = 0.5,
        percentile_norm: bool = True,
        pmin: float = 1.0,
        pmax: float = 99.8,
    ):
        transform_list = [ToFloat32()]

        if percentile_norm:
            transform_list.append(PercentileNormalize(pmin=pmin, pmax=pmax))
        else:
            transform_list.append(MinMaxNormalize())

        if gaussian_noise_std > 0:
            transform_list.append(
                AddGaussianNoise(mean=0.0, std=gaussian_noise_std, p=gaussian_noise_p)
            )

        if poisson_noise_scale > 0:
            transform_list.append(
                AddPoissonNoise(scale=poisson_noise_scale, p=poisson_noise_p)
            )

        if blur_kernel_size > 0 and blur_sigma > 0:
            transform_list.append(
                SpatialGaussianBlur(
                    kernel_size=blur_kernel_size, sigma=blur_sigma, p=blur_p
                )
            )

        if brightness_range and contrast_range:
            transform_list.append(
                RandomBrightnessContrast(
                    brightness_range=brightness_range,
                    contrast_range=contrast_range,
                    p=brightness_contrast_p,
                )
            )

        if intensity_scale_range:
            transform_list.append(
                RandomIntensityScaling(
                    scale_range=intensity_scale_range, p=intensity_scale_p
                )
            )

        if spatial_flip_p > 0:
            transform_list.append(
                RandomSpatialFlip(
                    flip_xy=True, flip_xz=True, flip_yz=True, p=spatial_flip_p
                )
            )

        if rotation_p > 0:
            transform_list.append(RandomSpatialRotation90(p=rotation_p))

        self.transforms = T.Compose(transform_list)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transforms(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class OneatTrainPresetHeavy:
    def __init__(
        self,
        gaussian_noise_std: float = 0.03,
        gaussian_noise_p: float = 0.6,
        poisson_noise_scale: float = 1.0,
        poisson_noise_p: float = 0.5,
        blur_kernel_size: int = 5,
        blur_sigma: float = 1.5,
        blur_p: float = 0.5,
        brightness_range: tuple = (0.8, 1.2),
        contrast_range: tuple = (0.8, 1.2),
        brightness_contrast_p: float = 0.7,
        intensity_scale_range: tuple = (0.9, 1.1),
        intensity_scale_p: float = 0.5,
        spatial_flip_p: float = 0.5,
        rotation_p: float = 0.5,
        elastic_alpha: float = 10.0,
        elastic_sigma: float = 3.0,
        elastic_p: float = 0.3,
        percentile_norm: bool = True,
        pmin: float = 1.0,
        pmax: float = 99.8,
    ):
        transform_list = [ToFloat32()]

        if percentile_norm:
            transform_list.append(PercentileNormalize(pmin=pmin, pmax=pmax))
        else:
            transform_list.append(MinMaxNormalize())

        if gaussian_noise_std > 0:
            transform_list.append(
                AddGaussianNoise(mean=0.0, std=gaussian_noise_std, p=gaussian_noise_p)
            )

        if poisson_noise_scale > 0:
            transform_list.append(
                AddPoissonNoise(scale=poisson_noise_scale, p=poisson_noise_p)
            )

        if blur_kernel_size > 0 and blur_sigma > 0:
            transform_list.append(
                SpatialGaussianBlur(
                    kernel_size=blur_kernel_size, sigma=blur_sigma, p=blur_p
                )
            )

        if brightness_range and contrast_range:
            transform_list.append(
                RandomBrightnessContrast(
                    brightness_range=brightness_range,
                    contrast_range=contrast_range,
                    p=brightness_contrast_p,
                )
            )

        if intensity_scale_range:
            transform_list.append(
                RandomIntensityScaling(
                    scale_range=intensity_scale_range, p=intensity_scale_p
                )
            )

        if elastic_alpha > 0 and elastic_sigma > 0:
            transform_list.append(
                ElasticDeformation(alpha=elastic_alpha, sigma=elastic_sigma, p=elastic_p)
            )

        if spatial_flip_p > 0:
            transform_list.append(
                RandomSpatialFlip(
                    flip_xy=True, flip_xz=True, flip_yz=True, p=spatial_flip_p
                )
            )

        if rotation_p > 0:
            transform_list.append(RandomSpatialRotation90(p=rotation_p))

        self.transforms = T.Compose(transform_list)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transforms(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class OneatEvalPreset:
    def __init__(
        self,
        percentile_norm: bool = True,
        pmin: float = 1.0,
        pmax: float = 99.8,
    ):
        transform_list = [ToFloat32()]

        if percentile_norm:
            transform_list.append(PercentileNormalize(pmin=pmin, pmax=pmax))
        else:
            transform_list.append(MinMaxNormalize())

        self.transforms = T.Compose(transform_list)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transforms(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


__all__ = [
    "OneatTrainPresetLight",
    "OneatTrainPresetMedium",
    "OneatTrainPresetHeavy",
    "OneatEvalPreset",
]

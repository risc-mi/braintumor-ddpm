import torch
import random
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F

__all__ = [
    "Compose",
    "CenterCrop",
    "RandomHorizontalFlip",
    "Resize",
    "PILToTensor",
    "ConvertImageDtype",
    "Lambda"]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        image = F.center_crop(image, self.size)
        if mask is not None:
            mask = F.center_crop(mask, self.size)
        return image, mask


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, mask):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if mask is not None:
                mask = F.hflip(mask)
        return image, mask


class RandomVerticalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, mask):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            if mask is not None:
                mask = F.vflip(mask)
        return image, mask


class RandomRotation:
    def __init__(self, rot_prob):
        self.rot_prob = rot_prob

    def __call__(self, image, mask):
        rotation_angle = np.random.uniform(low=-30, high=30)
        data_fill_value = torch.min(image).item()
        label_fill_value = torch.min(mask).item()

        if random.random() < self.rot_prob:
            image = F.rotate(img=image, angle=rotation_angle, fill=data_fill_value)
            mask = F.rotate(img=mask, angle=rotation_angle, fill=label_fill_value)

        return image, mask


class RandomAffine:
    def __init__(self, affine_prob):
        self.affine_prob = affine_prob

    def __call__(self, image, mask):
        # Apply affine transforms and sample affine values
        angle = np.random.uniform(low=-10, high=10, size=1).item()
        translate = list(np.random.uniform(low=-5, high=5, size=2))
        scale = np.random.uniform(low=0.85, high=1.15, size=1).item()
        shear = list(np.random.uniform(low=-3, high=3, size=2))

        if random.random() < self.affine_prob:
            # Apply transform
            image = F.affine(img=image,
                             angle=angle,
                             translate=translate,
                             scale=scale,
                             shear=shear,
                             fill=torch.min(image).item())

            mask = F.affine(img=mask,
                            angle=angle,
                            translate=translate,
                            scale=scale,
                            shear=shear,
                            fill=torch.min(mask).item())

        return image, mask


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        image = F.resize(image, self.size)
        if mask is not None:
            mask = F.resize(mask, self.size, interpolation=T.InterpolationMode.NEAREST)
        return image, mask


class RandomGamma:
    def __init__(self, gamma_prob):
        self.gamma_prob = gamma_prob

    def __call__(self, image, mask):

        gamma_value = np.random.uniform(low=0.5, high=1.25, size=1)
        if random.random() < self.gamma_prob:
            image[0, :, :] = F.adjust_gamma(image[0, :, :], gamma_value)
            image[1, :, :] = F.adjust_gamma(image[1, :, :], gamma_value)
            image[2, :, :] = F.adjust_gamma(image[2, :, :], gamma_value)
            image[3, :, :] = F.adjust_gamma(image[3, :, :], gamma_value)

        return image, mask


class RandomBrightness:
    def __init__(self, bright_prob):
        self.bright_prob = bright_prob

    def __call__(self, image, mask):
        bright_value = np.random.uniform(low=0.75, high=1.5, size=1)
        if random.random() < self.bright_prob:
            image[0, :, :] = F.adjust_gamma(image[0, :, :], bright_value)
            image[1, :, :] = F.adjust_gamma(image[1, :, :], bright_value)
            image[2, :, :] = F.adjust_gamma(image[2, :, :], bright_value)
            image[3, :, :] = F.adjust_gamma(image[3, :, :], bright_value)

        return image, mask


class PILToTensor:
    def __call__(self, image, mask):
        if not isinstance(image, torch.Tensor):
            image = F.pil_to_tensor(image)

        if not isinstance(mask, torch.Tensor) and mask is not None:
            mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
        return image, mask


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, mask):
        image = F.convert_image_dtype(image, self.dtype)
        return image


class Lambda:
    def __init__(self, lam):
        if not callable(lam):
            raise TypeError("argument should be callable.")
        self.lam = lam

    def __call__(self, image, mask):
        return self.lam(image), mask

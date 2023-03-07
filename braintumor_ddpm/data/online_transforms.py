import numpy as np
import torch
from typing import Tuple
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt

from braintumor_ddpm.data.datasets import SegmentationDataset


class OnlineAugmentation:
    def __init__(self, seed: int = 42):

        self.seed = seed

        # Flipping parameters
        self.p_hflip = 0.5
        self.p_vflip = 0.5

        # Rotation parameters
        self.p_rotate = 0.25
        self.angles = (-15, 15)

        # Affine parameters
        self.p_random_affine = 0.25
        self.affine_angle = (-5, 5)
        self.translate = (-4, 4)
        self.scale = (0.85, 1.10)
        self.shear = (-2, 2)

        # Functions
        self.functions = [self.random_horizontal_flip, self.random_vertical_flip,
                          self.random_rotation, self.random_affine]

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        image, mask = self.random_horizontal_flip(image, mask)
        image, mask = self.random_vertical_flip(image, mask)
        image, mask = self.random_rotation(image, mask)
        image, mask = self.random_affine(image, mask)
        return image, mask

    def random_horizontal_flip(self, data: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Applies random flipping based on a random probability """
        # Sample a probability value from a uniform distribution
        prob = torch.randn(1).item()

        # Apply flipping based on prob value
        if prob >= self.p_hflip:
            data = F.hflip(data)
            label = F.hflip(label)
        return data, label

    def random_vertical_flip(self, data: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Applies random flipping based on a random probability """
        # Sample a probability value from a uniform distribution
        prob = torch.randn(1).item()

        # Apply flipping based on prob value
        if prob >= self.p_vflip:
            data = F.vflip(data)
            label = F.vflip(label)
        return data, label

    def random_rotation(self, data: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Applies random rotation based on a random probability """

        # Sample a probability value from a uniform distribution
        prob = torch.randn(1).item()

        # Apply flipping based on prob value
        if prob >= self.p_hflip:
            # Sample angle from given range
            angle = np.random.uniform(low=self.angles[0], high=self.angles[1], size=1).item()
            data_fill_value = torch.min(data).item()
            label_fill_value = torch.min(label).item()
            data = F.rotate(img=data, angle=angle, fill=data_fill_value)
            label = F.rotate(img=label, angle=angle, fill=label_fill_value)
        return data, label

    def random_affine(self, data: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # Sample a probability value from a uniform distribution
        prob = torch.randn(1).item()

        if prob >= self.p_random_affine:
            # Apply affine transforms and sample affine values
            angle = np.random.uniform(low=self.affine_angle[0], high=self.affine_angle[1], size=1).item()
            translate = list(np.random.uniform(low=self.translate[0], high=self.translate[1], size=2))
            scale = np.random.uniform(low=self.scale[0], high=self.scale[1], size=1).item()
            shear = list(np.random.uniform(low=self.shear[0], high=self.shear[1], size=2))

            # Apply transform
            data = F.affine(img=data,
                            angle=angle,
                            translate=translate,
                            scale=scale,
                            shear=shear,
                            fill=torch.min(data).item())

            label = F.affine(img=label,
                             angle=angle,
                             translate=translate,
                             scale=scale,
                             shear=shear,
                             fill=torch.min(label).item())
        return data, label

    def random_brightness(self, data: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: implement additive brightness within limited range
        pass

    def random_gamma(self, data: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: implement gamma correction within a suitable range
        pass


# def plot_image_and_mask(data, label, data_aug, label_aug):
#     fig, ax = plt.subplots(2, 5, figsize=(10, 6))
#     titles = ["T1", "T1Gd", "T2", "Flair", "Ground Truth"]
#     for i in range(0, 5):
#         if i != 4:
#             ax[0][i].imshow(data[i, :, :], cmap="gray")
#             ax[1][i].imshow(data_aug[i, :, :], cmap="gray")
#             ax[0][i].set_title(titles[i], fontsize=14)
#         else:
#             ax[0][i].imshow(label)
#             ax[1][i].imshow(label_aug)
#             ax[0][i].set_title(titles[i], fontsize=14)
#         ax[0][i].set_axis_off()
#         ax[1][i].set_axis_off()
#     plt.show()


# dataset = SegmentationDataset(
#     images_dir=r"H:\BRATS\2D Stacked Images\scans",
#     masks_dir=r"H:\BRATS\2D Stacked Images\masks"
# )
#
# # Get image and apply augmentation
# idxs = [2, 8, 10, 76, 2093, 7881, 4672]
# for i in idxs:
#     image, mask = dataset[i]
#     online_augmentation = OnlineAugmentation()
#     img_aug, msk_aug = online_augmentation(image, mask)
#
#     plot_image_and_mask(image.numpy(),
#                         mask.squeeze().numpy(),
#                         img_aug.numpy(),
#                         msk_aug.squeeze().numpy())
#     plt.close()


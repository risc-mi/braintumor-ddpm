import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import random_split
from braintumor_ddpm.data.brats import BRATS
from braintumor_ddpm.data.datasets import SegmentationDataset
from braintumor_ddpm.utils.convert_data import move_data


# # TODO: add cmd arguments
# if __name__ == "__main__":
#     output_directory = input("Output directory: ")
#     path = input("BraTS 3D data: ")
#
#     # Extract 2D data from the original 3D dataset
#     brats_dataset = BRATS(path=path)
#     brats_dataset.export_stack(output_path=output_directory, slices=None)
#
#     # Create segmentation dataset
#     dataset = SegmentationDataset(images_dir=os.path.join(output_directory, "Stacked 2D BRATS Data", "scans"),
#                                   masks_dir=os.path.join(output_directory, "Stacked 2D BRATS Data", "masks"),
#                                   image_size=128,
#                                   device='cpu',
#                                   verbose=False)
#
#     # Split data into training pool and test pool
#     train_pool, test = random_split(dataset=dataset,
#                                     lengths=[757, 8000],
#                                     generator=torch.Generator().manual_seed(42))
#
#     print(f"training pool: {len(train_pool)} images, test data: {len(test)} images")







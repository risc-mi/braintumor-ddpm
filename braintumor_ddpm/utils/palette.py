import numpy as np
import torch
from .data import torch2np
from PIL import Image


def colorize(mask: torch.Tensor, palette: list):
    """

    """
    _mask = torch2np(mask, squeeze=True)
    _mask = Image.fromarray(_mask.astype(np.uint8)).convert('P')
    _mask.putpalette(palette)

    return np.array(_mask.convert('RGB'))


def get_palette(dataset: str):
    dataset = dataset.lower()
    if dataset == 'brats':
        return BRATS
    elif dataset == 'binary':
        return BRATS_BINARY
    else:
        raise ValueError(f"unknown palette {dataset}")


"""
Palette for datasets
currently using:
    1. Brats
    2. n/a
"""

BRATS = [
    45, 0, 55,  # 0: Background
    20, 90, 139,  # 1: Tumor core (BLUE)
    22, 159, 91,  # 2: Invaded Tissue (GREEN)
    255, 232, 9  # 3: Enhancing Tumor (YELLOW)

]

BRATS_BINARY = [
    45, 0, 55,  # 0: Background
    255, 232, 9  # 3: Whole Tumor (YELLOW)

]

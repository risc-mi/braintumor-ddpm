import json
import numpy as np
import torch
from torch.nn.functional import interpolate
from torch import nn
from sklearn.cluster import KMeans
from .data import torch2np


def save_tensors(module: nn.Module, tensors, name: str) -> None:
    """
    Saves tensors and sets [name] attribute to layers
    source: https://github.com/yandex-research/ddpm-segmentation
    """
    if type(tensors) in [list, tuple]:
        tensors = [t.detach().float() if t is not None else None for t in tensors]
        setattr(module, name, tensors)
    elif isinstance(tensors, dict):
        tensors = {k: t.detach().float() for k, t in tensors.items()}
        setattr(module, name, tensors)
    else:
        setattr(module, name, tensors.detach().float())


def save_inputs(self, inputs, outputs):
    """
    Saves input activations
    source: https://github.com/yandex-research/ddpm-segmentation
    """
    save_tensors(self, inputs[0], 'activations')
    return outputs


def save_outputs(self, inputs, outputs):
    """
    Saves output activations
    Source: https://github.com/yandex-research/ddpm-segmentation
    """
    save_tensors(self, outputs, 'activations')
    return outputs


def get_feature_clusters(x: torch.Tensor, output_size: int, clusters: int = 8):
    """ Applies KMeans across feature maps of an input activations tensor """
    if not isinstance(x, torch.Tensor):
        raise NotImplementedError(f"Function supports torch input tensors only, but got ({type(x)})")

    if x.ndim == 3:
        x = x.unsqueeze(0)

    b, c, h, w = x.shape
    assert h == w, f"image should be square, but got h = {h} and w = {w}"

    scale_factor = int(np.ceil(output_size / h))
    x = interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=True)
    x = torch2np(x, squeeze=True).reshape((output_size * output_size), c)
    x = KMeans(n_clusters=clusters, n_init='auto').fit_predict(x).reshape(output_size, output_size)
    return x

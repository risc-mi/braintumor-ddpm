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
    x = KMeans(n_clusters=clusters).fit_predict(x).reshape(output_size, output_size)
    return x


# TODO: review metrics2str() usage and delete
def metrics2str(mean_scores: dict, scores: dict, labels: list):
    """
    Converts given metrics to a string for display
    """
    metric_keys = list(scores.keys())
    mean_keys = list(mean_scores.keys())

    # output format
    out_str = ""
    for mean_metrics, metric in zip(mean_keys, metric_keys):
        out_str += f"{metric} scores: \n{'-' * 20}\n"
        out_str += f"mean: {mean_scores[mean_metrics].item():.3f} "
        for j, label in enumerate(labels):
            if len(labels) > 1:
                out_str += f"{label}: {scores[metric][j].item():.3f} "
            else:
                out_str += f"{label}: {scores[metric].item():.3f} "
        out_str += "\n\n"

    return out_str


# TODO: review read_json_metrics() usage and delete
def read_json_metrics(path: str, metrics: list = None, labels: list = None) -> dict:
    if metrics is None:
        metrics = ['dice', 'hd95', 'jaccard']
    if labels is None:
        labels = ['TC', 'IT', 'ET']

    with open(path, 'r') as jf:
        json_data = json.load(jf)
    jf.close()
    mean_metrics = {k: {label: [] for label in labels} for k in metrics}

    for key, val in json_data.items():
        for m in metrics:
            for label in labels:
                value = json_data[key][m][label]
                if value > 100:
                    value = float('nan')
                mean_metrics[m][label].append(value)

    for key, val in mean_metrics.items():
        for label in labels:
            mean_metrics[key][label] = np.nanmean(mean_metrics[key][label])
    return mean_metrics

import glob
import os
from typing import List, Optional, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from braintumor_ddpm.utils.data import torch2np, normalize
from braintumor_ddpm.utils.palette import colorize
import SimpleITK as sitk


def plot_modal(image: np.ndarray, fname: str, suptitle: str, titles: Optional[List[str]]):
    """ plot a single modal image """

    if titles is None:
        # we assume we are using brats
        titles = ["T1", "T1ce", "T2", "Flair"]

    # image shape in H W C B
    if image.ndim > 3:
        h, w, c, b = image.shape
        if b == 1:
            image = image.squeeze(-1)
        else:
            raise RuntimeError(f"plot_modal() supports only batch size of 1, got batch = {b}")
    elif image.ndim == 2:
        raise RuntimeError(f"expected channels > 1")

    h, w, c = image.shape
    fig, ax = plt.subplots(1, c, figsize=(8, 8))
    for i in range(0, c):
        ax[i].imshow(normalize(image)[:, :, i], cmap="gray")
        ax[i].set_axis_off()
        ax[i].set_title(titles[i])
    plt.suptitle(suptitle)
    plt.savefig(fname=fname, dpi=600)
    plt.close()


def plot_result(prediction: torch.Tensor,
                ground_truth: torch.Tensor,
                palette: list = None,
                file_name: str = None,
                title: str = None,
                caption: str = None,
                fontsize: int = 14):
    """
    Plots a prediction vs mask and optionally
    saves it to storage
    """
    fig, ax = plt.subplots(1, 2)
    if palette is None:
        ax[0].imshow(torch2np(prediction, squeeze=True))
        ax[1].imshow(torch2np(ground_truth, squeeze=True))
    else:
        ax[0].imshow(colorize(prediction, palette))
        ax[1].imshow(colorize(ground_truth, palette))

    ax[0].set_axis_off()
    ax[1].set_axis_off()

    ax[0].set_title("Prediction")
    ax[1].set_title("Ground Truth")

    if caption is not None:
        fig.text(0.5, 0.05, caption, ha='left', fontsize=fontsize)

    if title is not None:
        fig.suptitle(title, fontsize=fontsize)

    if file_name is not None:
        plt.savefig(file_name, dpi=600)
    plt.close()
    return fig, ax


def plot_debug(prediction: torch.Tensor,
               mask: torch.Tensor,
               images: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
               caption: str = None,
               file_name: str = None,
               title: str = None,
               palette: list = None,
               fontsize: int = 8):
    fig, ax = plt.subplots(4, 4)
    fig.subplots_adjust(hspace=0, wspace=0)

    titles = ["T1", "T1ce", "T2", "Flair"]
    image, noisy, denoised = images

    # show different modalities
    for i in range(0, image.shape[0]):

        # plot Original image as read from disk
        ax[0][i].imshow(torch2np(normalize(image)[i, :, :]), cmap="gray")
        ax[1][i].imshow(torch2np(normalize(noisy)[i, :, :]), cmap="gray")
        ax[2][i].imshow(torch2np(normalize(denoised)[i, :, :]), cmap="gray")

        ax[0][i].xaxis.set_major_locator(plt.NullLocator())
        ax[1][i].xaxis.set_major_locator(plt.NullLocator())
        ax[2][i].xaxis.set_major_locator(plt.NullLocator())
        ax[3][i].xaxis.set_major_locator(plt.NullLocator())

        ax[0][i].yaxis.set_major_locator(plt.NullLocator())
        ax[1][i].yaxis.set_major_locator(plt.NullLocator())
        ax[2][i].yaxis.set_major_locator(plt.NullLocator())
        ax[3][i].yaxis.set_major_locator(plt.NullLocator())

        ax[0][i].set_title(titles[i], fontsize=8)

        if i == 0:
            ax[0][i].set_ylabel("x", fontsize=8)
            ax[1][i].set_ylabel("x_noisy", fontsize=8)
            ax[2][i].set_ylabel("x_denoised", fontsize=8)
            ax[3][i].set_ylabel("GT\Predictions", fontsize=8)

        if i == 3:
            ax[i][0].imshow(colorize(prediction, palette))
            ax[i][1].imshow(colorize(mask, palette))

            ax[i][0].set_xlabel("Prediction", fontsize=8)
            ax[i][1].set_xlabel("Ground Truth", fontsize=8)

            ax[i][2].imshow(torch2np(normalize(image))[:, :, 0], cmap="gray")
            pred = colorize(prediction, palette)
            ax[i][2].imshow(np.ma.masked_where(pred == 0, pred), alpha=0.7)
            ax[i][2].set_xlabel("Prediction Overlay", fontsize=8)

            ax[i][3].imshow(torch2np(normalize(image))[:, :, 0], cmap="gray")
            gt = colorize(mask, palette)
            ax[i][3].imshow(np.ma.masked_where(gt == 0, gt), alpha=0.7)
            ax[i][3].set_xlabel("GT Overlay", fontsize=8)

    if file_name is not None:
        plt.savefig(file_name, dpi=800)
    plt.close()


def visualize_segmentations(images_folder: str,
                            ground_truth_folder: str,
                            predictions_folder: str,
                            baseline_folder: str,
                            output_folder: str,
                            modality_id: str = '0001') -> None:

    masks = os.listdir(ground_truth_folder)
    for mask in masks:

        # file name
        name = mask.split('.nii.gz')[0]
        try:
            # read image, mask, prediction and ground truth
            all_files = [
                glob.glob(os.path.join(images_folder, f"{name}_{modality_id}.*"))[0],
                os.path.join(ground_truth_folder, f"{name}.nii.gz"),
                os.path.join(predictions_folder, f"{name}.nii.gz"),
                os.path.join(baseline_folder, f"{name}.nii.gz"),
            ]

            all_images = []
            for file in all_files:
                image = sitk.ReadImage(file)
                image = sitk.GetArrayFromImage(image)
                all_images.append(image.squeeze(0))

            # plot results
            fig, ax = plt.subplots(1, 3)
            titles = ["Ground Truth", "Prediction", "Baseline"]
            for i in range(0, 3):
                ax[i].imshow(all_images[0], cmap="gray")
                ax[i].imshow(np.ma.masked_where(all_images[i + 1] == 0, all_images[i + 1]), alpha=0.65)
                ax[i].set_title(titles[i], fontsize=8)
                ax[i].set_axis_off()

            plt.savefig(os.path.join(output_folder, f"{name}.jpeg"), dpi=300)
            plt.close()

        except Exception as ex:
            raise RuntimeError(f"Error plotting file {name}\n\n"
                               f"Exception: {ex}")

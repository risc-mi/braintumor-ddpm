"""
Semantic Segmentation metrics as used by the nnUNet in:
https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/evaluation/metrics.py

Using the exact same approach for baseline comparability
"""
from typing import Tuple

import numpy as np
from medpy import metric
from braintumor_ddpm.insights.confusion_matrix import ConfusionMatrix


def dice(prediction: np.ndarray = None,
         reference: np.ndarray = None,
         confusion_matrix: ConfusionMatrix = None,
         nan_for_nonexisting: bool = True) -> float:
    """
    Calculates dice score using, where dice =  2TP / (2TP + FP + FN)
    Args:
        prediction: binary numpy array containing prediction
        reference: binary numpy array containing reference
        confusion_matrix: Confusion matrix object
        nan_for_nonexisting: Either to use NaN for non existing or zero

    Returns: dice score as a float

    """

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    pred_empty, pred_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if pred_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float((2 * tp) / ((2 * tp) + fp + fn))


def jaccard(prediction: np.ndarray = None,
            reference: np.ndarray = None,
            confusion_matrix: ConfusionMatrix = None,
            nan_for_nonexisting: bool = True) -> float:
    """
    Calculates jaccard score, or IoU defined as iou = TP / (TP + FP + FN)
    Args:
        prediction: binary numpy array containing prediction
        reference: binary numpy array containing reference
        confusion_matrix: Confusion matrix object
        nan_for_nonexisting:

    Returns: Jaccard score or IoU as a float
    """

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    pred_empty, pred_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if pred_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp + fn))


def precision(prediction: np.ndarray = None,
              reference: np.ndarray = None,
              confusion_matrix: ConfusionMatrix = None,
              nan_for_nonexisting: bool = True) -> float:
    """
    Calculates precision, defined as TP/(TP + FP)
    Args:
        prediction: binary numpy array containing prediction
        reference: binary numpy array containing reference
        confusion_matrix: Confusion matrix object
        nan_for_nonexisting:

    Returns: Jaccard score or IoU as a float
    """

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    pred_empty, pred_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if pred_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp))


def sensitivity(prediction: np.ndarray = None,
                reference: np.ndarray = None,
                confusion_matrix: ConfusionMatrix = None,
                nan_for_nonexisting: bool = True) -> float:
    """
    Calculates sensitivity, defined as TP/(TP + FN)
    Args:
        prediction: binary numpy array containing prediction
        reference: binary numpy array containing reference
        confusion_matrix: Confusion matrix object
        nan_for_nonexisting:

    Returns: sensitivity score as a float

    """

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    pred_empty, pred_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fn))


def specificity(prediction: np.ndarray = None,
                reference: np.ndarray = None,
                confusion_matrix: ConfusionMatrix = None,
                nan_for_nonexisting: bool = True) -> float:
    """
    Calculates specificity, defined as TN/(TN + FP)
    Args:
        prediction: binary numpy array containing prediction
        reference: binary numpy array containing reference
        confusion_matrix: Confusion matrix object
        nan_for_nonexisting:

    Returns: sensitivity score as a float
    """

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    pred_empty, pred_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tn / (tn + fp))


def hausdorff_distance(prediction: np.ndarray = None,
                       reference: np.ndarray = None,
                       confusion_matrix: ConfusionMatrix = None,
                       nan_for_nonexisting: bool = True,
                       voxel_spacing: Tuple = None,
                       connectivity: int = 1) -> float:
    """
    Calculates specificity, defined as TN/(TN + FP)
    Args:

        prediction: binary numpy array containing prediction
        reference: binary numpy array containing reference
        confusion_matrix: Confusion matrix object
        nan_for_nonexisting:
        voxel_spacing:
        connectivity:

    Returns: sensitivity score as a float
    """

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    pred_empty, pred_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if pred_empty or pred_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    prediction, reference = confusion_matrix.prediction, confusion_matrix.reference
    return metric.hd(prediction, reference, voxel_spacing, connectivity)


def hausdorff_distance_95(prediction: np.ndarray = None,
                          reference: np.ndarray = None,
                          confusion_matrix: ConfusionMatrix = None,
                          nan_for_nonexisting: bool = True,
                          voxel_spacing: Tuple = None,
                          connectivity: int = 1) -> float:
    """
    Calculates specificity, defined as TN/(TN + FP)
    Args:

        prediction: binary numpy array containing prediction
        reference: binary numpy array containing reference
        confusion_matrix: Confusion matrix object
        nan_for_nonexisting:
        voxel_spacing:
        connectivity:

    Returns: sensitivity score as a float
    """

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    pred_empty, pred_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if pred_empty or pred_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    prediction, reference = confusion_matrix.prediction, confusion_matrix.reference
    return metric.hd95(prediction, reference, voxel_spacing, connectivity)


ALL_METRICS = {
    "Dice": dice,
    "Jaccard": jaccard,
    "Precision": precision,
    "Sensitivity": sensitivity,
    "Specificity": specificity,
    "Hausdorff Distance": hausdorff_distance,
    "Hausdorff Distance 95": hausdorff_distance_95
}

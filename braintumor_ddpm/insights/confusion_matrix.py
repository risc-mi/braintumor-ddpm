"""
Based on nnUNet metrics.py and modified a bit ofr our use case
https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/evaluation/metrics.py
"""
import numpy as np


class ConfusionMatrix:
    """
    Evaluates binary images and calculates confusion matrix and other metrics
    for semantic segmentation evaluation
    """

    def __init__(self, prediction: np.ndarray, reference: np.ndarray) -> None:

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.prediction = None
        self.reference = None
        self.reference_empty = None
        self.reference_full = None
        self.pred_empty = None
        self.pred_full = None

        # set prediction/reference
        self.set_reference(reference)
        self.set_prediction(prediction)

    def set_prediction(self, prediction) -> None:
        self.prediction = prediction
        self.reset()

    def set_reference(self, reference) -> None:
        self.reference = reference
        self.reset()

    def reset(self):
        """ Resets Confusion Matrix entries """
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.pred_empty = None
        self.pred_full = None

    @staticmethod
    def assert_shape(prediction, reference):

        assert prediction.shape == reference.shape,\
            f"Shape mismatch between {prediction.shape} and {reference.shape}"

    def compute(self):
        """ Computes confusion matrix for current reference/prediction """
        if self.prediction is None or self.reference is None:
            raise ValueError(f"'prediction' and 'reference' must be set prior to computation")

        self.assert_shape(self.prediction, self.reference)

        # Calculate confusion matrix
        self.tp = int(((self.prediction != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.prediction != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.prediction == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.prediction == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.pred_empty = not np.any(self.prediction)
        self.pred_full = np.all(self.prediction)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):
        """ Returns the calculated confusion matrix """
        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):
        """ Returns size of reference array """
        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):
        """ Existence of prediction/ reference as in full array or empty arrays """
        for case in (self.pred_empty, self.pred_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break
        return self.pred_empty, self.pred_full, self.reference_empty, self.reference_full





import os
import glob
import json
from collections import OrderedDict
from typing import Tuple
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

from braintumor_ddpm.insights.cm_metrics import ALL_METRICS
from braintumor_ddpm.insights.confusion_matrix import ConfusionMatrix


class NiftiEvaluator:
    def __init__(self, predictions: str,
                 references: str,
                 labels: dict = None) -> None:
        """
        Evaluator class, takes in two directories and evaluates them against specific metrics.
        Both directories should contain the same file names, where files should be nifti files
        (.nii.gz)

        Args:
            predictions (str): A string pointing to the directory exported prediction files.
            references (str): A string  pointing to the directory of ground truth files.
        """
        self.predictions = predictions
        self.references = references
        self.labels = labels
        self.regions = {
            "Whole Tumor": (1, 2, 3),
            "Enhancing Tumor": (3,),
            "Tumor Core": (2, 3)
        }

        if self.labels is None:
            raise ValueError(f"a labels dictionary should be supplied to evaluate corresponding labels!")

        # Check prediction files against references
        self.all_preds = os.listdir(self.predictions)
        self.all_refs = os.listdir(self.references)

        if len(self.all_preds) != len(self.all_refs):
            raise RuntimeError(f"Mismatch between references and predictions {len(self.all_preds)} != {len(self.all_refs)}")

        for pred, ref in zip(self.all_preds, self.all_refs):
            if os.path.basename(pred) != os.path.basename(ref):
                raise RuntimeError(f"File name mismatch {os.path.basename(pred)} != {os.path.basename(ref)}")
        print(f"\nChecked predictions and references directory successfully..\n")

    @staticmethod
    def load_file(pred_path: str, ref_path: str) -> Tuple:
        """
        Reads both nifti files for a prediction and a reference image, then returns them as numpy arrays
        Args:
            pred_path: path to prediction file
            ref_path: path to reference file

        Returns: Tuple of numpy arrays of prediction and reference images
        """
        prediction = sitk.ReadImage(pred_path)
        prediction = sitk.GetArrayFromImage(prediction)

        reference = sitk.ReadImage(ref_path)
        reference = sitk.GetArrayFromImage(reference)
        return prediction.squeeze(), reference.squeeze()

    def evaluate_file(self, pred_path: str, ref_path: str) -> OrderedDict:
        """
        Evaluates a single prediction against a reference
        Args:
            pred_path: path to prediction file
            ref_path: path to reference file

        Returns: Dictionary of evaluated metrics
        """

        # read files into numpy arrays
        prediction, reference = self.load_file(pred_path=pred_path, ref_path=ref_path)
        result = OrderedDict()

        if isinstance(self.labels, dict):
            # calculate metrics for each label
            for name, label in self.labels.items():

                # get binary masks for current label
                prediction_mask = np.where(prediction == label, 1, 0)
                reference_mask = np.where(reference == label, 1, 0)
                result[name] = {}

                confusion_matrix = ConfusionMatrix(prediction_mask, reference_mask)
                for key, metric in ALL_METRICS.items():
                    result[name][key] = metric(
                        prediction=prediction_mask,
                        reference=reference_mask,
                        confusion_matrix=confusion_matrix)

        elif isinstance(self.labels, list):
            # calculate metrics for each label
            for label in self.labels:

                # get binary masks for current label
                prediction_mask = np.where(prediction == label, 1, 0)
                reference_mask = np.where(reference == label, 1, 0)
                result[label] = {}

                confusion_matrix = ConfusionMatrix(prediction_mask, reference_mask)
                for key, metric in ALL_METRICS.items():
                    result[label][key] = metric(
                        prediction=prediction_mask,
                        reference=reference_mask,
                        confusion_matrix=confusion_matrix)
        del prediction, reference, prediction_mask, reference_mask, confusion_matrix
        return result

    def evaluate_file_regions(self, pred_path: str, ref_path: str) -> OrderedDict:
        """
        Evaluates a single prediction against a reference
        Args:
            pred_path: path to prediction file
            ref_path: path to reference file

        Returns: Dictionary of evaluated metrics
        """

        # read files into numpy arrays
        prediction, reference = self.load_file(pred_path=pred_path, ref_path=ref_path)
        result = OrderedDict()

        # calculate metrics for each label
        for name, region in self.regions.items():
            prediction_mask = np.isin(prediction, region)
            reference_mask = np.isin(reference, region)
            confusion_matrix = ConfusionMatrix(prediction_mask, reference_mask)
            result[name] = {}

            for key, metric in ALL_METRICS.items():
                result[name][key] = (metric(
                    prediction=prediction_mask,
                    reference=reference_mask,
                    confusion_matrix=confusion_matrix))

        del prediction, reference, prediction_mask, reference_mask, confusion_matrix
        return result

    def evaluate_folders(self, output_dir: str = None, brats_regions: bool = True) -> OrderedDict:
        """
        Evaluates all files in predictions directory against references directory
        Returns: Ordered Dictionary for all files and their corresponding metrics
        """
        all_results = OrderedDict()
        all_results_regions = OrderedDict()
        mean_labels = {label: {m: [] for m in ALL_METRICS.keys()} for label in self.labels.keys()}
        mean_regions = {label: {m: [] for m in ALL_METRICS.keys()} for label in self.regions.keys()}

        with tqdm(zip(self.all_preds, self.all_refs), total=len(self.all_preds)) as pbar:
            for pred, ref in pbar:
                # get filenames
                pred_filename = os.path.basename(pred)
                ref_file_path = glob.glob(os.path.join(self.references, pred_filename))[0]
                pred_file_path = os.path.join(self.predictions, pred_filename)

                # calculate scores
                if brats_regions:
                    scores_regions = self.evaluate_file_regions(pred_path=pred_file_path, ref_path=ref_file_path)
                scores = self.evaluate_file(pred_path=pred_file_path, ref_path=ref_file_path)

                # append region based results
                if brats_regions:
                    all_results_regions[pred_filename] = scores_regions

                    # append to mean dictionary
                    for region in self.regions.keys():
                        for metric in ALL_METRICS.keys():
                            mean_regions[region][metric].append(scores_regions[region][metric])

                # append label based results
                all_results[pred_filename] = scores
                for label in self.labels.keys():
                    for metric in ALL_METRICS.keys():
                        mean_labels[label][metric].append(scores[label][metric])

        # compute mean for each metric entry (regions)
        if brats_regions:
            for region in self.regions.keys():
                for metric in ALL_METRICS.keys():
                    mean_regions[region][metric] = np.nanmean(mean_regions[region][metric])
            all_results_regions['mean'] = mean_regions

        # compute mean for each metric entry (separate labels)
        for label in self.labels.keys():
            for metric in ALL_METRICS.keys():
                mean_labels[label][metric] = np.nanmean(mean_labels[label][metric])

        # append mean to results dict
        all_results['mean'] = mean_labels

        print(f"Finished evaluating directories.")
        if output_dir is not None:
            with open(os.path.join(output_dir, "summary_labels.json"), 'w') as jf:
                json.dump(all_results, jf, indent=4)
            jf.close()

            with open(os.path.join(output_dir, "summary_regions.json"), 'w') as jf:
                json.dump(all_results_regions, jf, indent=4)
            jf.close()

        return all_results


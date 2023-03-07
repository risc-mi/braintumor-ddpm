import os
import glob
from typing import Union

import torch
import numpy as np
from PIL import Image
import tifffile as tiff
import SimpleITK as itk
from torch.utils.data import Dataset
from braintumor_ddpm.utils.data import torch2np, normalize


class BRATS(Dataset):
    """
    A PyTorch Dataset utilized for Brain Tumor Segmentation Dataset
    """

    def __init__(self, path: str, validation: bool = False) -> None:
        """
        Initializes the dataset using a path to all patient folders
        :param path (str): directory containing all patient scans
        """
        self.path = path
        self.val = validation
        self.dataset = dict()

        # check for cuda
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Make sure path exists
        if not os.path.exists(self.path):
            raise RuntimeError("Given path ({}) does not exist.".format(self.path))

        # Path should be a directory to all MRI cases
        if not os.path.isdir(self.path):
            raise NotADirectoryError("Given path ({}) is not a directory".format(self.path))

        # List all sub folders within path, exclude files
        _folders = os.listdir(self.path)
        _folders = [os.path.join(self.path, item) for item in _folders]
        _folders = [item for item in _folders if os.path.isdir(item)]
        _total_items = len(_folders)

        for index, folder in enumerate(_folders):
            folder_name = os.path.basename(folder)

            # Ensure every folder has 5 items/files
            if len(os.listdir(folder)) != 5 and not self.val:
                raise FileNotFoundError("One or more missing files in folder: {}.\n"
                                        "Check contents {}".format(folder_name, os.listdir(folder)))

            # Get path to all MRI modalities and segmentation
            t1 = glob.glob(os.path.join(folder, "{}_t1.*".format(folder_name)))[0]
            t1ce = glob.glob(os.path.join(folder, "{}_t1ce*".format(folder_name)))[0]
            t2 = glob.glob(os.path.join(folder, "{}_t2*".format(folder_name)))[0]
            flair = glob.glob(os.path.join(folder, "{}_flair*".format(folder_name)))[0]
            if not self.val:
                label = glob.glob(os.path.join(folder, "{}_seg*".format(folder_name)))[0]

            # TODO: make sure all files are for the same patient

            # Save path for every MRI modality to dataset
            if self.val:
                self.dataset[index] = {
                    '3d': {
                        "t1": t1,
                        "t1ce": t1ce,
                        "t2": t2,
                        "flair": flair,
                    }
                }
            else:
                self.dataset[index] = {
                    '3d': {
                        "t1": t1,
                        "t1ce": t1ce,
                        "t2": t2,
                        "flair": flair,
                        "label": label
                    }
                }
            print("\rReading item(s) [{}/{}]".format(index, _total_items - 1), end="", flush=False)
        print("\nFinished Fetching all dataset items")

    def __len__(self) -> int:
        return len(self.dataset.keys())

    def __getitem__(self, index) -> dict:
        """ Reads and returns data in 3D """

        data = self.dataset[index]['3d']

        try:
            # MRI modalities
            t1 = itk.ReadImage(data['t1'])
            t1ce = itk.ReadImage(data['t1ce'])
            t2 = itk.ReadImage(data['t2'])
            flair = itk.ReadImage(data['flair'])
            label = None
            if not self.val:
                label = itk.ReadImage(data['label'])

            # TODO: Torch can not cast uint16 to tensors, check labels after casting!
            if self.val:
                data = {
                    't1': torch.tensor(itk.GetArrayFromImage(t1), dtype=torch.float32).to(self.device),
                    't1ce': torch.tensor(itk.GetArrayFromImage(t1ce), dtype=torch.float32).to(self.device),
                    't2': torch.tensor(itk.GetArrayFromImage(t2), dtype=torch.float32).to(self.device),
                    'flair': torch.tensor(itk.GetArrayFromImage(flair), dtype=torch.float32).to(self.device),
                }

                data = {
                    'volume': torch.cat([
                        data['t1'].unsqueeze(0),
                        data['t1ce'].unsqueeze(0),
                        data['t2'].unsqueeze(0),
                        data['flair'].unsqueeze(0)
                    ])
                }

            else:
                data = {
                    't1': torch.tensor(itk.GetArrayFromImage(t1), dtype=torch.float32).to(self.device),
                    't1ce': torch.tensor(itk.GetArrayFromImage(t1ce), dtype=torch.float32).to(self.device),
                    't2': torch.tensor(itk.GetArrayFromImage(t2), dtype=torch.float32).to(self.device),
                    'flair': torch.tensor(itk.GetArrayFromImage(flair), dtype=torch.float32).to(self.device),
                    'label': torch.tensor(itk.GetArrayFromImage(label).astype(dtype=np.int32),
                                          dtype=torch.int32).to(self.device)
                }
                data = {
                    'volume': torch.cat([
                        data['t1'].unsqueeze(0),
                        data['t1ce'].unsqueeze(0),
                        data['t2'].unsqueeze(0),
                        data['flair'].unsqueeze(0)
                    ]),
                    'label': data['label'].unsqueeze(0)
                }

        except Exception as ex:
            raise RuntimeError("unable to read data at index: {} in dataset.\nError:{}".format(index, ex))

        return data

    def export_stack(self,
                     output_path: str,
                     slices: Union[list, str] = None):
        """
        Exports 2D stacked multi-page tiff files for given slices from 3D BraTS data

        :param output_path: output directory
        :param slices: a list of slices to extract, if all slices are to be extracted use slices='all'.
                        defaults to {70, 75,..., 95, 100}
        :return: writes the extracted slices to the given output directory
        """

        # define output folders
        output_folder = os.path.join(output_path, 'Stacked 2D BRATS Data', 'scans')
        if not self.val:
            labels_folder = os.path.join(output_path, 'Stacked 2D BRATS Data', 'masks')
            if not os.path.exists(labels_folder):
                os.makedirs(labels_folder)

        # create directories even if it does not exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Generate stacked 2d images across every slice index
        if slices is None:
            slices = [70, 75, 80, 85, 90, 95, 100]

        if isinstance(slices, str):
            if slices.lower() == 'all':
                slices = [i for i in range(0, 155)]

        total = len(self.dataset)

        # extension based on either tensor or not
        _ext = "tiff"
        for index in range(0, total):

            # Get different MRI scans
            data = self.__getitem__(index)
            name = os.path.basename(self.dataset[index]['3d']['t1']).split('_t1')[0]
            # t1 = data['t1']
            # t1ce = data['t1ce']
            # t2 = data['t2']
            # flair = data['flair']
            volume = data['volume']
            if not self.val:
                label = data['label']

            # Export stacked images per slices
            for i in slices:
                # stack slices
                stacked = volume[:, i, :, :]

                # Save images/masks in tiff with original format
                out_file = os.path.join(output_folder, "{}_{}.{}".format(name, i, _ext))
                tiff.imwrite(out_file, normalize(torch2np(stacked)))
                if not self.val:
                    label_file = os.path.join(labels_folder, "{}_{}.{}".format(name, i, _ext))
                    tiff.imwrite(label_file, torch2np(label[:, i, :, :]))
                print("\rExporting scan [{}/{}]".format((index + 1), total), end='', flush=True)

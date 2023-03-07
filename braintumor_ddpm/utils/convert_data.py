""" Based on https://github.com/MIC-DKFZ/nnUNet conversion to 2D files, modified and extended for BraTS-2D """
import os
import torch
import json
import shutil
import numpy as np
import SimpleITK as sitk
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Optional, Union, Tuple
from batchgenerators.utilities.file_and_folder_operations import save_json, subfiles


def _convert_brats_id(name: str, slice_id: int) -> str:
    """ Takes BraTS2021_12345.nii.gz and converts it to BraTS_12345sSLICE """
    name_copy = name
    name_copy = name_copy.split('_')[-1]
    name_copy = f"BraTS_{int(name_copy):05d}s{slice_id:03d}"
    return name_copy


def _convert_brats_labels(segmentation: np.ndarray) -> np.ndarray:
    """ Convert BraTS labels to continuous ones """

    seg_copy = np.zeros_like(segmentation)
    seg_copy[segmentation == 4] = 3
    seg_copy[segmentation == 1] = 2
    seg_copy[segmentation == 2] = 1

    return seg_copy


def _resize_volume(volume: np.ndarray, size: int = 128,
                   method: T.InterpolationMode = T.InterpolationMode.BILINEAR) -> np.ndarray:
    # copy volume and convert to a torch tensor

    volume_copy = np.copy(volume)
    volume_copy = torch.from_numpy(volume_copy)
    volume_copy = F.resize(volume_copy, size=[size], interpolation=method)

    return volume_copy.numpy()


def _create_task_folder(root: str, task_id: int, task_name: str):
    """ creates a Task folder within root directory """
    task_folder = os.path.join(root, f"Task{task_id}_{task_name}")
    os.makedirs(task_folder, exist_ok=True)

    # define directories
    images_tr = os.path.join(task_folder, "imagesTr")
    labels_tr = os.path.join(task_folder, "labelsTr")
    images_ts = os.path.join(task_folder, "imagesTs")
    labels_ts = os.path.join(task_folder, "labelsTs")

    # create dirs
    os.makedirs(images_tr, exist_ok=True)
    os.makedirs(labels_tr, exist_ok=True)
    os.makedirs(images_ts, exist_ok=True)
    os.makedirs(labels_ts, exist_ok=True)

    return task_folder, images_tr, labels_tr, images_ts, labels_ts


def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques


def move_data(target_dir: str,
              split: dict,
              images_dir: str,
              labels_dir: str,
              modality_mapping: dict = None,
              seed: int = 16,
              task_id: int = 501
              ):
    """
    Args:
        task_id:
        seed:
        target_dir:
        split:
        images_dir:
        labels_dir:
        modality_mapping:

    Returns:
    """
    if modality_mapping is None:
        modality_mapping = {
            't1': '0000',
            't1ce': '0001',
            't2': '0002',
            'flair': '0003'
        }

    # create task folders
    task_folder, images_tr, labels_tr, images_ts, labels_ts = _create_task_folder(root=target_dir,
                                                                                  task_name=f"BraTS{seed}",
                                                                                  task_id=task_id)
    print(f"\nCopying the exported slices to task folder")

    for entry in split['training']:
        images = os.path.join(images_dir, entry)
        images = [f"{images}_{modality_mapping[key]}.nii.gz" for key in modality_mapping.keys()]

        # copy image files and modalities
        for im_file in images:
            shutil.copy(
                src=os.path.join(images_dir, im_file),
                dst=images_tr
            )

        # copy label file
        shutil.copy(
            src=os.path.join(labels_dir, f"{entry}.nii.gz"),
            dst=labels_tr
        )

    for entry in split['testing']:
        name = os.path.join(images_dir, entry)
        images = [f"{name}_{modality_mapping[key]}.nii.gz" for key in modality_mapping.keys()]

        # copy images and modalities
        for im_file in images:
            shutil.copy(
                src=os.path.join(images_dir, im_file),
                dst=images_ts
            )

        # copy labels
        shutil.copy(
            src=os.path.join(labels_dir, f"{entry}.nii.gz"),
            dst=labels_ts
        )

    generate_dataset_json(
        output_file=os.path.join(task_folder, "dataset.json"),
        imagesTr_dir=images_tr,
        imagesTs_dir=images_ts,
        modalities=tuple(modality_mapping.keys()),
        labels={
            0: 'background',
            1: 'edema',
            2: 'non-enhancing',
            3: 'enhancing'
        },
        dataset_name=f'BraTS2021_s{seed}'
    )


def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                          labels: dict, dataset_name: str, sort_keys=True, license: str = "hands off!",
                          dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param sort_keys: In order to sort or not, the keys in dataset.json
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i
        in
        train_identifiers]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file), sort_keys=sort_keys)


def convert_brats_to_2d(dataset_path: str,
                        target_dir: str,
                        slices: list = None,
                        split: Optional[Union[dict, str]] = None,
                        size: int = 128) -> None:
    """
    Converts BraTS2021 dataset from 3D to 2D format across different axial slices, and
    prepares it to be directly used with an nnUNet or similar structure to Medical Segmentation Decathlon

    As 2D slices are exported, we change the identifier for each case to be XXXXXsYYY, where XXXXX is the patient ID
    in the original dataset and sYYY indicates the slice ID from that patient.

    For example: a BraTS2021_00089.nii.gz sliced at index 75 will generate BraTS_00089s075.nii.gz file
    the date is removed to be consistent with nnUNet usage

    """
    # validate dataset_path
    if dataset_path is None or not isinstance(dataset_path, str):
        raise TypeError(f"expected dataset_path to be of type string but got {type(dataset_path)}")

    if not os.path.exists(dataset_path):
        raise FileExistsError(f"dataset_path [{dataset_path}] does not exist")

    try:
        # try to create target_dir
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        # create imagesTr directory
        images_dir = os.path.join(target_dir, "all_images")
        os.makedirs(images_dir, exist_ok=True)

        # create labelsTr directory
        labels_dir = os.path.join(target_dir, "all_labels")
        os.makedirs(labels_dir, exist_ok=True)

    except Exception as ex:
        raise RuntimeError(f"failed at creating target_dir [{target_dir}]\n{ex}")

    print(f"\nCreated all sub-directories in target folder"
          f" ({target_dir}) successfully")

    # constants and defaults for BraTS dataset
    if slices is None or len(slices) == 0:
        _slices = [70, 75, 80, 85, 90, 95, 100]
        slices = []
        for i in range(0, 155, 5):
            if i not in _slices:
                slices.append(i)

    # check data split, otherwise use all for training
    if isinstance(split, str):
        with open(split, 'r') as jf:
            split = json.load(jf)
        jf.close()
        print(f"\nUsing a predefined split")
    else:
        if split is None or len(list(split.keys())) == 0:
            all_data = os.listdir(dataset_path)
            all_data = [os.path.join(dataset_path, d) for d in all_data]
            split = {
                'training': all_data,
                'testing': []
            }
            print(f"\nUsing all data for training, no split file passed or configured")

    # helpers for naming and moving files around
    modalities = ['flair', 'seg', 't1', 't1ce', 't2']
    modality_mapping = {
        't1': '0000',
        't1ce': '0001',
        't2': '0002',
        'flair': '0003'
    }
    total_cases = len(os.listdir(dataset_path))
    print(f"\nTotal cases found: {total_cases}\n"
          f"Training Images: {len(split['training'])}\n"
          f"Testing Images: {len(split['testing'])}\n"
          f"{'=' * 40}\n")

    # go over each patient, extract all slices and save them to target directory
    print(f"\nExtracting all slices and labels")
    for i, patient in enumerate(os.listdir(dataset_path)):
        for m in modalities:
            # read each modality and check the file exists
            case_id = os.path.basename(patient)
            case_id = os.path.join(dataset_path, patient, f"{case_id}_{m}.nii.gz")

            if not os.path.exists(case_id):
                raise FileNotFoundError(f"file {case_id} was not found!")
            else:
                # read volume
                volume = sitk.ReadImage(fileName=case_id)
                volume = sitk.GetArrayFromImage(volume)

                # maintain dtypes for consistency
                if m == 'seg':
                    volume = volume.astype(dtype=np.uint8)
                    volume = _convert_brats_labels(volume)
                    interpolation_method = T.InterpolationMode.NEAREST
                    file_path = labels_dir
                else:
                    volume = volume.astype(dtype=np.int16)
                    interpolation_method = T.InterpolationMode.BILINEAR
                    file_path = images_dir

                # resize the volume accordingly and return to sitk.Image
                volume = _resize_volume(volume=volume, size=size, method=interpolation_method)

                for s in slices:
                    axial_slice = volume[s, :, :]
                    axial_slice = sitk.JoinSeries(sitk.GetImageFromArray(axial_slice))
                    axial_slice.SetOrigin([0, 0, 0])
                    axial_slice.SetSpacing([1, 1, 999])

                    # filename
                    name = os.path.basename(patient)
                    name = _convert_brats_id(name, s)
                    name = f"{name}_{modality_mapping[m]}.nii.gz" if m != 'seg' else f"{name}.nii.gz"
                    file_name = os.path.join(file_path, name)
                    sitk.WriteImage(image=axial_slice, fileName=file_name)
                print(
                    f"\rExporting slices from patient"
                    f" {os.path.basename(patient)}, progress [{i + 1}/{total_cases}]", end='')

    # create task folders
    move_data(target_dir=target_dir,
              split=split,
              images_dir=images_dir,
              labels_dir=labels_dir,
              modality_mapping=modality_mapping)

    print(f"\nFinished exporting and splitting data")


import torch
import blobfile as bf
from typing import Tuple, Union, Callable
import torchvision.transforms as Tr
from torch.utils.data import Dataset
import braintumor_ddpm.data.transforms as t
from braintumor_ddpm.utils.data import imread, brats_labels


class SegmentationDataset(Dataset):
    """
    A dataset object for semantic segmentation in Pytorch, requires path to images and
    masks folders

    :param images_dir: directory pointing to image data as a string
    :param masks_dir: directory pointing to mask data as a string
    :param image_size: final image/mask size after transformations
    :param seed: random seed
    :param device: a string representing which device either cpu/cuda
    """

    def __init__(self,
                 images_dir: str,
                 masks_dir: str,
                 image_size: int = 128,
                 transforms: Union[t.Compose, Tr.Compose] = None,
                 seed: int = 42,
                 device: str = 'cpu',
                 process_labels: Callable = None,
                 train: bool = True,
                 verbose: bool = True) -> None:

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.seed = seed
        self.device = device
        self.transforms = transforms
        self.verbose = verbose

        # labels processing
        if process_labels is None:
            self.map_labels = brats_labels
        else:
            self.map_labels = process_labels

        self.dataset = []
        self.train = train
        self._ext = ['jpg', 'jpeg', 'tif', 'tiff', 'png']

        # set seed manually
        torch.manual_seed(self.seed)

        # define transforms
        if self.transforms is None:
            self.transforms_train = t.Compose([
                t.Resize(self.image_size),
                t.CenterCrop(self.image_size),
                t.RandomHorizontalFlip(0.5),
                t.PILToTensor(),
                t.Lambda(lambda v: (v * 2) - 1)
            ])
        else:
            self.transforms_train = self.transforms

        self.transforms_test = t.Compose([
                t.Resize(self.image_size),
                t.CenterCrop(self.image_size),
                t.PILToTensor(),
                t.Lambda(lambda v: (v * 2) - 1)
            ])

        if self.train:
            self.transforms = self.transforms_train
        else:
            self.transforms = self.transforms_test

        # validate directories
        if not bf.exists(self.images_dir):
            raise FileExistsError("images directory does not exits")
        if self.train:
            if not bf.exists(self.masks_dir):
                raise FileExistsError("masks directory does not exits")

        all_images = bf.listdir(self.images_dir)
        all_images = [bf.join(self.images_dir, img) for img in all_images if img.split('.')[-1].lower() in self._ext]
        if self.train:
            all_masks = bf.listdir(self.masks_dir)
            all_masks = [bf.join(self.masks_dir, msk) for msk in all_masks if msk.split('.')[-1].lower() in self._ext]

            if len(all_images) != len(all_masks):
                raise RuntimeError(f"total images ({len(all_images)}) does not match total masks ({len(all_masks)})")
        i = 0
        image_name, mask_name = "", ""
        try:
            # attempt to create a dataset of images and masks
            for i in range(0, len(all_images)):

                # get image and mask names
                image_name = bf.basename(all_images[i]).split('.')[0].lower()
                if self.train:
                    mask_name = bf.basename(all_masks[i]).split('.')[0].lower()

                    # image name and mask name should be equivalent
                    if image_name != mask_name:
                        raise NameError(f"image ({image_name}) and mask ({mask_name}) names are not matching")

                if self.train:
                    # add items to dataset
                    self.dataset.append({
                        'image': all_images[i],
                        'mask': all_masks[i]
                    })
                else:
                    self.dataset.append({
                        'image': all_images[i]
                    })
                if self.verbose:
                    print(f"\rCreating segmentation dataset [{i + 1}/{len(all_images)}]", end='', flush=True)
            print(f"\rCreated segmentation dataset with {len(all_images)} items\n")

        except Exception as ex:
            raise RuntimeError(f"error occurred while creating dataset at index ({i})\n"
                               f"Image name {image_name}\n"
                               f"Mask name {mask_name}\n"
                               f"Error: {ex}")

    def __len__(self):
        return len(self.dataset)

    def set_test_transforms(self):
        self.transforms = self.transforms_test

    def set_train_transforms(self):
        self.transforms = self.transforms_train

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:

        # get data at given index
        data = self.dataset[item]
        if self.train:
            image_path, mask_path = data['image'], data['mask']
        else:
            image_path = data['image']

        # read image and mask
        image = imread(image_path)

        if self.train:
            mask = imread(mask_path)
        else:
            mask = None

        # apply transforms
        image, mask = self.transforms(image, mask)

        if self.train:
            mask = self.map_labels(mask)
            return image.to(self.device), mask.to(self.device).long()
        else:
            return image.to(self.device)


class PixelDataset(Dataset):
    """
    Dataset class containing all pixel representations/features and
    their corresponding labels

    :param x_data: a flattened tensor with all pixels activations with a shape of (num_pixels, num_features)
    :param y_data: a flattened tensor with all pixel labels with a shape of (num_pixels)
    """

    def __init__(self,
                 x_data: torch.Tensor,
                 y_data: torch.Tensor,
                 device: str = 'cuda') -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.device = device

    def __len__(self) -> int:
        return len(self.x_data)

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        """ returns a single pixel representation and it's target label """
        return self.x_data[item].to(self.device), self.y_data[item].to(self.device)


class ImageDataset(Dataset):
    def __init__(self,
                 images_dir: str,
                 image_size: int = 128,
                 transforms: Union[t.Compose, Tr.Compose] = None,
                 seed: int = 42,
                 device: str = 'cpu'
                 ):
        self.images_dir = images_dir
        self.image_size = image_size
        self.transforms = transforms
        self.seed = seed
        self.device = device

        self.extensions = ['tiff', 'tif', 'jpeg', 'jpg', 'png']
        self.dataset = []

        # set seed
        torch.manual_seed(self.seed)

        # check path
        if not bf.exists(self.images_dir):
            raise FileExistsError(f"given directory ({self.images_dir}) does not exist")

        if not bf.isdir(self.images_dir):
            raise NotADirectoryError(f"given path ({self.images_dir}) is not a directory")

        if self.transforms is None:
            self.transforms = Tr.Compose([
                Tr.Resize(self.image_size),
                Tr.RandomHorizontalFlip(0.5),
                Tr.CenterCrop(self.image_size),
                # Tr.ToTensor(),  # our imread directly returns a tensor
                Tr.Lambda(lambda v: (v * 2) - 1)
            ])
        try:
            self.dataset = [bf.join(self.images_dir, img) for img in bf.listdir(self.images_dir)
                            if img.split('.')[-1].lower() in self.extensions]
        except Exception as ex:
            raise RuntimeError("Unable to create image dataset.\n"
                               f"Error: {ex}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_path = self.dataset[index]
        image = imread(image_path)
        image = self.transforms(image)
        return image, {}

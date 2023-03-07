import json
import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import braintumor_ddpm.data.transforms as t
import SimpleITK as sitk
from torch.optim import lr_scheduler
from braintumor_ddpm.utils.data import torch2np
from braintumor_ddpm.insights.cm_metrics import dice
from braintumor_ddpm.insights.plots import plot_result
from torch.utils.data import random_split, DataLoader, Dataset
from braintumor_ddpm.data.datasets import SegmentationDataset, PixelDataset
from braintumor_ddpm.data.online_transforms import OnlineAugmentation
from braintumor_ddpm.core.training.SimpleNetworkTrainer import SimpleNetworkTrainer


class DenoiseNetworkTrainer(SimpleNetworkTrainer):
    def __init__(self,
                 network: nn.Module,
                 output_folder: str,
                 images_dir: str,
                 labels_dir,
                 train_size: int,
                 train_pool_size: int = 757,
                 validation_size: int = 250,
                 test_size: int = 8000,
                 use_validation_from_train: bool = False,
                 batch_size: int = 64,
                 seed: int = 16,
                 device: str = 'cpu',
                 labels: dict = None,
                 experiment_id: str = None
                 ) -> None:
        super(DenoiseNetworkTrainer, self).__init__()

        # Parameters to training and network
        self.device = device
        self.network = network
        self.batch_size = batch_size
        self.lr_patience = 10
        self.softmax = nn.Softmax(dim=1)
        self.loss_function = nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.3, 0.3, 0.3]).to(self.device))

        # Dataset related variables/params
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.train_size = train_size
        self.train_pool_size = train_pool_size
        self.validation_size = validation_size
        self.use_validation_from_train = use_validation_from_train
        self.test_size = test_size
        self.labels = labels
        self.split_metadata = {'train': {
            'ids': None,
            'files': []
        },
            'valid': {
                'ids': None,
                'files': []
            },
            'test': {
                'ids': None,
                'files': []
            }
        }

        # Variables/params for pipeline or experiment settings
        self.image_size = None
        self.output_folder = output_folder
        self.experiment_id = experiment_id
        self.seed = seed

    def initialize(self) -> None:
        """ Initialization of the entire pipeline """

        self.setup_folders()
        self.initialize_network()

        # Set image size after initializing network
        self.image_size = self.network.config["image_size"]
        self.initialize_data_loaders()
        self.initialize_optimizer_and_scheduler()
        self.write_to_log_file(content=f"Initialized Pipeline ..\n")

    def initialize_network(self) -> None:
        self.network.initialize()
        self.network.set_device(self.device)
        self.is_initialized = True

    def initialize_data_loaders(self) -> None:
        """ Creates all relevant data loaders for training """
        self.write_to_log_file(content=f"Preparing data splits and data-loaders ..\n")
        # First create a SemanticSegmentation Dataset
        self.dataset = SegmentationDataset(images_dir=self.images_dir,
                                           masks_dir=self.labels_dir,
                                           image_size=self.image_size,
                                           transforms=t.Compose([
                                               t.Resize(self.image_size),
                                               t.CenterCrop(self.image_size),
                                               t.RandomHorizontalFlip(0.2),
                                               t.RandomVerticalFlip(0.2),
                                               t.RandomRotation(0.2),
                                               t.RandomAffine(0.1),
                                               t.RandomGamma(0.2),
                                               t.RandomBrightness(0.1),
                                               t.PILToTensor(),
                                               t.Lambda(lambda v: (v * 2) - 1)]),
                                           device=self.device)

        # Split to training pool and test data
        training_pool, self.test_data = random_split(dataset=self.dataset,
                                                     lengths=[self.train_pool_size, self.test_size],
                                                     generator=torch.Generator().manual_seed(42))

        # Threshold validation size if it's more than available data
        resample_validation = False
        expected_validation_size = self.train_pool_size - self.train_size

        # In case passed validation_size is not the same with available data
        if self.validation_size != expected_validation_size:

            # either more than available data -> threshold to max. available data
            if self.validation_size > expected_validation_size:
                self.validation_size = expected_validation_size
            else:
                # resample from available data
                resample_validation = True

        self.train_data, self.valid_data = random_split(
            dataset=training_pool,
            lengths=[self.train_size, expected_validation_size],
            generator=torch.Generator().manual_seed(self.seed))

        if resample_validation:
            self.valid_data, _ = random_split(
                dataset=self.valid_data,
                lengths=[self.validation_size, (expected_validation_size - self.validation_size)],
                generator=torch.Generator().manual_seed(self.seed)
            )
            del _
        if self.use_validation_from_train:
            self.valid_data = self.train_data

        # Setup data-loaders
        self.train_dl = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True)
        self.valid_dl = DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=True)

        # Save metadata/split information
        if len(self.test_data) > 0:
            for index in self.test_data.indices:
                filename = os.path.basename(self.dataset.dataset[index]['mask'])
                self.split_metadata['test']['files'].append(filename)
            self.split_metadata['test']['ids'] = self.test_data.indices

        if len(self.train_data) > 0:
            for index in self.train_data.indices:
                filename = os.path.basename(self.dataset.dataset[index]['mask'])
                self.split_metadata['train']['files'].append(filename)
            self.split_metadata['train']['ids'] = self.train_data.indices

        if len(self.valid_data) > 0:
            for index in self.valid_data.indices:
                filename = os.path.basename(self.dataset.dataset[index]['mask'])
                self.split_metadata['valid']['files'].append(filename)
            self.split_metadata['valid']['ids'] = self.valid_data.indices

        with open(os.path.join(self.output_folder, "data_splits.json"), 'w') as jf:
            json.dump(self.split_metadata, jf)

    def initialize_optimizer_and_scheduler(self) -> None:
        """ Setup optimizer and optionally the lr scheduler """
        self.write_to_log_file(content=f"Initializing optimizer and lr_scheduler\n")
        try:
            if self.is_initialized:
                # create and configure optimizer
                self.optimizer = torch.optim.Adam(
                    params=self.network.denoise_network.parameters(),
                    lr=self.initial_lr,
                    amsgrad=True
                )

                # create lr_scheduler
                self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=0.95, patience=self.lr_patience, min_lr=self.minimum_lr
                )
        except Exception as ex:
            self.write_to_log_file(content=f"Error during initializing optimizer and lr_scheduler."
                                           f"\n\n{ex}\n", print_to_console=False)
            raise RuntimeError(f"Error during initializing optimizer and lr_scheduler.\n\n{ex}\n")

    def on_epoch_end(self) -> bool:
        """ Overwritten from parent class to include online epoch to epoch augmentation """
        # Plot progress and optionally update learning rate
        self.plot_progress()
        self.update_learning_rate()

        # Save checkpoint
        if self.epoch % self.save_every == 0:
            self.save_checkpoint(filename=f"checkpoint_epoch_{self.epoch}.pt")

        # Check for early stopping and manage training patience
        continue_training = self.manage_training_patience()

        return continue_training

    def run_evaluation(self, predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
        """ Runs evaluation on input batches """
        predictions = torch.argmax(self.softmax(predictions), dim=1)
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        batch_scores = np.zeros(shape=(predictions.shape[0], len(self.labels.keys())))
        for batch in range(0, predictions.shape[0]):
            pred, target = predictions[batch], targets[batch]

            for name, label in self.labels.items():
                mask_gt = np.where(target == label, 1, 0)
                mask_pred = np.where(pred == label, 1, 0)
                score = dice(mask_pred.squeeze(), mask_gt.squeeze())
                batch_scores[batch, label] = score

        return batch_scores

    def manage_training_patience(self) -> bool:
        """ Manages training patience and early stopping """
        return True

    def predict_test_data(self, save_predictions: str = 'random') -> None:
        """
        Performs predictions on test data
        Args:
            save_predictions (str): Can be 'random', 'all' or 'None', when 'all' is selected the entire data will be saved
                                    as plots of prediction vs. targets. While, 'random' selects a random percentage of
                                     entire data to be saved. Lastly 'None' does not save any plots.
        """

        # Make sure no transforms are applied to test data
        self.network.eval()
        self.dataset.set_test_transforms()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.network.set_device('cuda')

        # Setup proper paths to predictions folders
        self.predictions_output_folder = os.path.join(self.output_folder, "Predictions")
        nifti_folder = os.path.join(self.predictions_output_folder, "NIFTI Predictions")
        plots_folder = os.path.join(self.predictions_output_folder, "Plots")

        # Create folders
        os.makedirs(nifti_folder, exist_ok=True)
        os.makedirs(plots_folder, exist_ok=True)

        if save_predictions.lower() == 'random':
            selected_indices = np.random.choice(a=self.test_data.indices, size=int(0.05 * len(self.test_data)))
        elif save_predictions.lower() == 'all':
            selected_indices = self.test_data.indices
        else:
            selected_indices = []

        with tqdm(enumerate(self.test_data.indices), total=len(self.test_data)) as pbar:
            for i, idx in pbar:

                # TODO: delete self.test_dl if not in use
                # Get item from dataset within self.test_data
                data, targets = self.dataset[idx]
                data, targets = data.to(self.device).unsqueeze(0), targets.to(self.device)

                # Get filename to save it as is
                filename = os.path.basename(self.dataset.dataset[idx]['mask'])
                filename = filename.split('.tif')[0]
                filename = filename.split('_')
                filename = f"BraTS_{int(filename[1]):05d}s{int(filename[2]):03d}"

                # Predict on data
                prediction = self.network(data)
                prediction = torch.argmax(self.softmax(prediction), dim=1)

                if len(selected_indices) > 0 and idx in selected_indices:
                    plot_result(
                        prediction=prediction,
                        ground_truth=targets,
                        palette=[45, 0, 55,  # 0: Background
                                 20, 90, 139,  # 1: Non Enhancing (BLUE)
                                 22, 159, 91,  # 2: Tumor Core (GREEN)
                                 255, 232, 9  # 3: Enhancing Tumor (YELLOW)
                                 ],
                        file_name=os.path.join(plots_folder, f"{filename}.jpeg")
                    )

                # Save prediction as a nifti file
                prediction = torch2np(prediction, squeeze=True).astype(np.uint8)
                prediction = sitk.JoinSeries(sitk.GetImageFromArray(prediction))
                # print(prediction.GetSize())
                prediction.SetOrigin([0, 0, 0])
                prediction.SetSpacing([1, 1, 999])
                sitk.WriteImage(image=prediction, fileName=os.path.join(nifti_folder, f"{filename}.nii.gz"))

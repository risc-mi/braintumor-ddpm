"""
A very basic and simple Network Trainer Class
Modified/inspired from nnUNet v1:
https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/nnunet/training/network_training/network_trainer.py
"""

import os
from time import time
from datetime import datetime
from abc import abstractmethod
from typing import Tuple, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler


class SimpleNetworkTrainer:
    def __init__(self) -> None:
        """
        A very simple, avoiding any complicated approaches to training a network
        """

        # Core parameters essential for training
        self.network = None
        self.optimizer = None
        self.loss_function = None
        self.lr_scheduler = None
        self.is_initialized = False

        # Values/params used for data
        self.train_data = self.valid_data = self.test_data = None
        self.train_dl = self.valid_dl = self.test_dl = None

        # Running metrics, losses and others
        self.dataset = None
        self.train_losses, self.valid_losses = [], []
        self.train_eval_metrics, self.valid_eval_metrics = [], []
        self.all_lrs = []
        self.calculate_training_metrics = False
        self.calculate_validation_metrics = True

        # Patience related variables
        self.best_train_epoch = None
        self.best_validation_epoch = None

        # Output folders and directories
        self.experiment_id = None
        self.log_file = self.output_folder = None
        self.predictions_output_folder = None

        # Pipeline training settings
        self.train = True
        self.device = 'cpu'
        self.epoch = 0
        self.seed = 16
        self.maximum_epochs = 250
        self.initial_lr = 0.001
        self.minimum_lr = 1e-06
        self.lr_patience = 5
        self.save_every = 100
        self.validate_every = -1
        self.labels = None
        self.predict_test_after_training = True

    def set_initial_lr(self, lr: float) -> None:
        """ Changes the default lr """
        self.initial_lr = lr

    def set_validate_every(self, validate_every: int) -> None:
        self.validate_every = validate_every

    def set_maximum_epochs(self, max_epochs: int) -> None:
        """ Changes default self.maximum_epochs """
        self.maximum_epochs = max_epochs

    def write_to_log_file(self, content: str, print_to_console: bool = True, use_time_stamp: bool = False):
        """ Writes events/ content to log file located in the output folder """

        current_timestamp = datetime.now()
        timestamp_str = current_timestamp.strftime("%d/%m/%Y %H:%M:%S")

        if self.log_file is None:
            # create a new log-file
            filename = f"log_file_{current_timestamp.strftime('%d_%m_%Y_%H_%M_%S')}.txt"

            # make sure output folder exists/ otherwise create it
            if self.output_folder is None:
                self.setup_folders()

            self.log_file = os.path.join(self.output_folder, filename)
            with open(self.log_file, 'w') as f:
                if use_time_stamp:
                    f.write(f"Created log file successfully at {timestamp_str}\n\n")
                else:
                    f.write(f"Created log file successfully\n\n")

        # Write contents to logfile
        try:
            if use_time_stamp:
                update_message = f"{timestamp_str}\n" \
                                 f"{'==' * 25}\n" \
                                 f"{str(content)}"
            else:
                update_message = f"{str(content)}"

            # append update message to file
            with open(self.log_file, 'a') as f:
                f.write(update_message)

            if print_to_console:
                print(update_message)

        except Exception as ex:
            raise RuntimeError(f"Unable to write the following content to log file!\n"
                               f"content: {content}\n"
                               f"See Error:\n{ex}")

    def save_checkpoint(self, filename: str, save_optimizer: bool = True) -> None:
        """ Saves a checkpoint to output folder with the given filename """
        network_state_dict = self.network.state_dict()
        network_state_dict = {key: value.cpu() for key, value in network_state_dict.items()}

        lr_scheduler_state_dict = None
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'state_dict'):
            lr_scheduler_state_dict = self.lr_scheduler.state_dict()

        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
        else:
            optimizer_state_dict = None

        data_to_save = {
            "epoch": self.epoch,
            "network_state_dict": network_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "lr_scheduler_state_dict": lr_scheduler_state_dict,
            "losses": (self.train_losses, self.valid_losses),
            "metrics": (self.train_eval_metrics, self.valid_eval_metrics),
            "all_learning_rates": self.all_lrs
        }
        torch.save(data_to_save, os.path.join(self.output_folder, filename))
        self.write_to_log_file(content=f"Saved checkpoint to {os.path.join(self.output_folder, filename)}\n")

    def load_checkpoint(self, filename: str) -> None:
        """ Loads a checkpoint from disk and moves it to declared device """
        if not self.is_initialized:
            self.initialize()

        checkpoint = torch.load(filename, map_location=self.device)

        self.network.load_state_dict(checkpoint["network_state_dict"])
        if self.train:
            if checkpoint["lr_scheduler_state_dict"] is not None:
                if self.lr_scheduler is not None and hasattr(self.lr_scheduler, "load_state_dict"):
                    self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

            if checkpoint["optimizer_state_dict"] is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"] + 1
        self.train_losses, self.valid_losses = checkpoint["losses"]
        self.train_eval_metrics, self.valid_eval_metrics = checkpoint["metrics"]
        self.all_lrs = checkpoint["all_learning_rates"]
        self.write_to_log_file(f"Loaded Checkpoint successfully.\n")

    @abstractmethod
    def initialize(self) -> None:
        """ Initializes the entire Pipeline and sets up all relevant folders """

    @abstractmethod
    def initialize_network(self) -> None:
        """ Initializes self.network, its weights and all relevant parameters """

    @abstractmethod
    def initialize_data_loaders(self) -> None:
        """ Initializes all data loaders and generators """

    @abstractmethod
    def initialize_optimizer_and_scheduler(self) -> None:
        """ Initializes the optimizer used for learning and possible the lr_scheduler """

    def setup_folders(self) -> None:
        """ Sets up output folders """

        if self.output_folder is None:
            print(f"output_folder was not configured, using current directory: {os.getcwd()}")
            self.output_folder = os.getcwd()

        # setup output folder name
        if self.experiment_id is None:
            self.output_folder = os.path.join(self.output_folder,
                                              f"{self.__class__.__name__}",
                                              f"Experiment Seed {self.seed}")
        else:
            self.output_folder = os.path.join(self.output_folder,
                                              f"{self.__class__.__name__}",
                                              f"{self.experiment_id}")

        # Create folder, even if it exists
        os.makedirs(self.output_folder, exist_ok=True)

    def run_training(self) -> None:
        """ Runs the entire training procedure """

        if not self.is_initialized:
            raise RuntimeError(f"Initialize Pipeline first before training!")

        if self.device != 'cpu':
            if not torch.cuda.is_available():
                self.write_to_log_file(content=f"CUDA is not available and device is {self.device}\n")
            else:
                torch.cuda.empty_cache()

        """
        ==================================================
        ================ Started Training ================
        """
        self.write_to_log_file(content=f"\n\n\n{'=' * 14}\tStarted Training\t{'=' * 14}\n\n")

        while self.epoch < self.maximum_epochs:

            # Print current epoch and timestamp
            self.write_to_log_file(content=f"Epoch {self.epoch}:\n", use_time_stamp=True)
            epoch_start = time()
            batch_training_losses, batch_training_scores = [], []

            # set network to training mode
            self.network.train()
            for data, targets in self.train_dl:
                loss, scores = self.run_iteration(data, targets, run_backpropagation=True,
                                                  run_evaluation=self.calculate_training_metrics)
                if scores is not None:
                    batch_training_scores.append(scores)
                batch_training_losses.append(loss)

            # append batch level losses/scores to epoch level by calculating the mean
            self.train_losses.append(np.nanmean(batch_training_losses))
            if len(batch_training_scores) > 0:
                self.train_eval_metrics.append(np.nanmean(batch_training_scores))

            # possibly run validation step
            if (self.epoch % self.validate_every == 0) and self.validate_every != -1:
                # set network to evaluation mode
                self.network.eval()
                batch_validation_losses, batch_validation_scores = [], []
                for data, targets in self.valid_dl:
                    loss, scores = self.run_iteration(data, targets, run_backpropagation=False,
                                                      run_evaluation=self.calculate_validation_metrics)
                    if scores is not None:
                        batch_validation_scores.append(scores)
                    batch_validation_losses.append(loss)
                self.valid_losses.append(np.nanmean(batch_validation_losses))
                if len(batch_validation_scores) > 0:
                    self.valid_eval_metrics.append(np.nanmean(batch_validation_scores))

            # write to log and possibly console current losses/metrics
            content = f"training loss: {self.train_losses[-1]:.5f}"
            if self.validate_every != -1 and len(self.valid_losses) > 0:
                content += f", validation loss: {self.valid_losses[-1]:.5f}\n"
            self.write_to_log_file(content=content)

            continue_training = self.on_epoch_end()
            if not continue_training:
                self.write_to_log_file(content=f"Early stopping!\n", use_time_stamp=True)

            # update epoch and write total running time
            self.epoch += 1
            epoch_end = time()
            epoch_time = epoch_end - epoch_start
            self.write_to_log_file(content=f"Epoch {self.epoch - 1} took {epoch_time: .3f}s to run\n\n")
        self.save_checkpoint(filename="final_model.pt")

        # Optionally predict on the test dataset
        if self.predict_test_after_training:
            self.predict_test_data()

    def run_iteration(self, data: torch.Tensor, targets: torch.Tensor, run_backpropagation: bool,
                      run_evaluation: bool) -> Tuple[float, Optional[float]]:
        """ Runs a single iteration and optionally performs backpropagation or evaluation """

        batch_loss, batch_scores = None, None
        # Move to GPU if device is not cpu and cuda is available
        if self.device != 'cpu':
            if torch.cuda.is_available():
                data = data.cuda()
                targets = targets.cuda()

        # Run a forward pass
        self.optimizer.zero_grad()
        predictions = self.network(data)
        del data

        # Calculate loss
        batch_loss = self.loss_function(predictions, targets.squeeze())

        # Optionally run backpropagation
        if run_backpropagation:
            batch_loss.backward()
            self.optimizer.step()

        # Optionally evaluate
        if run_evaluation:
            batch_scores = self.run_evaluation(predictions=predictions, targets=targets)
            batch_scores = batch_scores[:,  1:]
            batch_scores = np.nanmean(batch_scores, axis=0)

        # Delete unused tensors
        del predictions, targets

        return batch_loss.detach().cpu().numpy(), batch_scores

    def update_learning_rate(self) -> None:
        # TODO: Check learning rate updates properly
        """ From the name, updates learning rate based on the lr_scheduler """
        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(self.train_losses[-1], epoch=self.epoch)
            else:
                self.lr_scheduler.step(self.epoch)
            self.all_lrs.append(self.optimizer.param_groups[0]['lr'])
            self.write_to_log_file(f"Current lr: {self.optimizer.param_groups[0]['lr']}")

    @abstractmethod
    def run_evaluation(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """ Runs an evaluation on input data and reports relevant metrics """
        pass

    @abstractmethod
    def predict_test_data(self) -> None:
        """ Predicts the data in self.test_data and saves it to self.predictions_output_folder"""

    @abstractmethod
    def manage_training_patience(self) -> bool:
        """ Manages early stopping and saves best epochs to output folder """
        pass

    def plot_progress(self) -> None:
        """ Plots current progress to progress.jpeg in the output folder """

        # Create figure and an extra twin y-axis
        fig, ax = plt.subplots(figsize=(20, 15))
        all_plots = []
        twin1 = ax.twinx()

        # Actual plotting
        labels = ["Training Loss", "Validation Loss", "Validation Metrics", "Training Metrics"]
        train_epochs = [j for j in range(0, self.epoch + 1)]
        validation_epochs = [j for j in range(0, self.epoch + 1, self.validate_every)]

        # First plot, training losses
        plot1, = ax.plot(train_epochs, self.train_losses, label=labels[0])
        all_plots.append(plot1)

        # Second plot, validation losses
        plot2, = ax.plot(validation_epochs, self.valid_losses, label=labels[1])
        all_plots.append(plot2)

        # Optional plots if existing
        if len(self.valid_eval_metrics) > 0:
            plot3, = twin1.plot(validation_epochs, self.valid_eval_metrics, label=labels[2], color='r')
            all_plots.append(plot3)

        if len(self.train_eval_metrics) > 0:
            plot4, = twin1.plot(train_epochs, self.train_eval_metrics, label=labels[3], color='g')
            all_plots.append(plot4)

        # Plot settings
        ax.grid(alpha=0.25)
        ax.set_title("Training Progress", fontsize=14)
        ax.set_xlabel("Epochs", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        twin1.set_ylabel("Evaluation Metric", fontsize=14)
        twin1.set_ylim([0, 1.0])
        ax.legend(handles=all_plots, loc="upper left")

        # Saving figure
        plt.savefig(os.path.join(self.output_folder, "progress.jpeg"), dpi=600)
        plt.close()

    def on_epoch_end(self) -> bool:
        """ Functions to call by the end of an epoch """

        # Plot progress and optionally update learning rate
        self.plot_progress()
        self.update_learning_rate()

        # Save checkpoint
        if self.epoch % self.save_every == 0:
            self.save_checkpoint(filename=f"checkpoint_epoch_{self.epoch}.pt")

        # Check for early stopping and manage training patience
        continue_training = self.manage_training_patience()

        return continue_training

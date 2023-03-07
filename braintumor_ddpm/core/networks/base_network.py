""" Basic Network class for all Diffusion based segmentation models """
import torch
from torch import nn
from abc import abstractmethod
from braintumor_ddpm.diffusion.config import brats_128x128_config
from braintumor_ddpm.scripts.script_utils import load_model


class GenericDiffusionNetwork(nn.Module):
    def __init__(self):
        """
        A generic Denoising Diffusion Probabilistic Model network that uses the ddpm based representations
        to learn mappings to targets.
        """
        super(GenericDiffusionNetwork, self).__init__()

        # Diffusion params/variables
        self.diffusion, self.denoise_network = None, None
        self.is_ddpm_initialized = False

        # Classifier and representations params/variables
        self.feature_classifier = None
        self.config = None

    def get_device(self) -> str:
        """ Returns current device """
        if next(self.parameters()).device.type == 'cpu':
            return 'cpu'
        else:
            return next(self.parameters()).device

    def set_device(self, device) -> None:
        """ Moves all modules to given device """
        if device == 'cpu':
            self.cpu()
        else:
            if isinstance(device, str):
                self.cuda()
            elif isinstance(device, int):
                self.cuda(device)

    def load_diffusion_model(self, model_path: str, config: dict = None) -> None:
        """ Loads a pre-trained diffusion model """
        if config is None:
            config = brats_128x128_config()
            self.config = config

        # load denoise network and diffusion model
        self.denoise_network, self.diffusion = load_model(model_path=model_path, config=config)
        self.denoise_network.eval()
        self.denoise_network = self.denoise_network.cpu()
        self.is_ddpm_initialized = True

    @abstractmethod
    def get_configuration(self) -> dict:
        pass

    @abstractmethod
    def load_configuration(self, config: dict) -> None:
        pass

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

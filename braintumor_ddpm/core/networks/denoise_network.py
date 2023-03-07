import torch
from torch import nn
from braintumor_ddpm.core.networks.base_network import GenericDiffusionNetwork
from braintumor_ddpm.diffusion.improved_ddpm.nn_utils import zero_module, conv_nd


class DenoiseNetwork(GenericDiffusionNetwork):

    def __init__(self,
                 ddpm_model_path: str,
                 time_steps: list,
                 num_classes: int,
                 freeze_encoder: bool = True):
        super().__init__()

        self.time_steps = time_steps
        self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=1)

        # Network related stuff
        self.ddpm_model_path = ddpm_model_path
        self.freeze_encoder = freeze_encoder

    def get_configuration(self) -> dict:
        configuration = {
            'time_steps': self.time_steps,
            'num_classes': self.num_classes,
            'freeze_encoder': self.freeze_encoder
        }
        return configuration

    def load_configuration(self, config: dict) -> None:
        """ Loads a configuration dictionary into parameters """
        self.time_steps = config["time_steps"]
        self.num_classes = config["num_classes"]

    def initialize(self) -> None:

        # First load diffusion model into module
        self.load_diffusion_model(model_path=self.ddpm_model_path)

        # Change output layer
        self.denoise_network.out = nn.Sequential(
            nn.SiLU(),
            zero_module(conv_nd(self.denoise_network.dims,
                                self.denoise_network.input_chs,
                                self.num_classes,
                                3,
                                padding=1))
        )
        if self.freeze_encoder:
            self.denoise_network.freeze_encoder()

    def feature_extractor(self, x: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Extracts features from denoise network,
        No need to extract features in current network
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Has mainly two modes, if in inference a full image representations is passed,
        otherwise a batch random pixels to train on
        """
        time = torch.tensor(self.time_steps).to(self.denoise_network.device)
        x_noisy = self.diffusion.q_sample(x_start=x, t=time, noise=None)
        x = self.denoise_network(x_noisy, time)

        return x

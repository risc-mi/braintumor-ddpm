""" Pixel Level Classifier class that extends GenericDiffusionNetwork """

import torch
from torch import nn
from braintumor_ddpm.utils.data import scale_features
from braintumor_ddpm.utils.helpers import save_inputs, save_outputs
from braintumor_ddpm.core.networks.base_network import GenericDiffusionNetwork


class PixelClassifier(GenericDiffusionNetwork):

    def __init__(self,
                 ddpm_model_path: str,
                 layers: list,
                 time_steps: list,
                 num_classes: int = 4,
                 use_input_activations: bool = False):
        super().__init__()

        self.layers = layers
        self.time_steps = time_steps
        self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=1)

        # Feature extractor stuff
        self.feature_layers = []
        self.feature_classifier = None
        self.in_features = None
        self.ddpm_model_path = ddpm_model_path
        self.use_input_activations = use_input_activations
        self.save_hook = save_inputs if self.use_input_activations else save_outputs

    def get_configuration(self) -> dict:
        configuration = {
            'layers': self.layers,
            'time_steps': self.time_steps,
            'num_classes': self.num_classes,
            'in_features': self.in_features,
            'use_input_activations': self.use_input_activations
        }
        return configuration

    def load_configuration(self, config: dict) -> None:
        """ Loads a configuration dictionary into parameters """
        self.layers = config["layers"]
        self.time_steps = config["time_steps"]
        self.num_classes = config["num_classes"]
        self.in_features = config["in_features"]
        self.use_input_activations = config["use_input_activations"]

    def initialize(self) -> None:

        # First load diffusion model into module
        self.load_diffusion_model(model_path=self.ddpm_model_path)

        # Add hooks to extract representations from denoise network
        for idx, layer in enumerate(self.denoise_network.output_blocks):
            if idx in self.layers:
                layer.register_forward_hook(self.save_hook)
                self.feature_layers.append(layer)

        if self.in_features is None:
            self.in_features = self._calculate_input_features()

        # Setup feature classifier
        self.feature_classifier = nn.Sequential(
            nn.Linear(self.in_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(64, self.num_classes)
        )

    def _calculate_input_features(self) -> int:
        """ Compute the number of input features """

        channels = self.config.get("in_channels")
        image_size = self.config.get("image_size")
        sample_image = torch.randn(size=(1, channels, image_size, image_size)).to(self.get_device())
        features = self.feature_extractor(sample_image)
        in_features = features.shape[1]

        del channels, image_size, sample_image, features
        return in_features

    def feature_extractor(self, x: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """ Extracts features from denoise network """
        activations = []
        for t in self.time_steps:

            # Create a time tensor
            t = torch.tensor([t]).to(self.get_device())

            if x.ndim == 3:
                x = x.unsqueeze(0)
            x = x.to(self.get_device())
            noisy_x = self.diffusion.q_sample(x_start=x, t=t, noise=noise)
            self.denoise_network(noisy_x, t)

            for layer in self.feature_layers:
                activations.append(layer.activations)
                layer.activations = None

        activations = scale_features(activations=activations, size=self.config.get("image_size"))
        return activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Has mainly two modes, if in inference a full image representations is passed,
        otherwise a batch random pixels to train on
        """
        if self.feature_classifier.training:
            # In training mode, representations already extracted (training approach)
            x = self.feature_classifier(x)
        else:
            # In inference mode, we extract representations first
            x = self.feature_extractor(x)
            b, f, h, w = x.shape

            # change to (batch, features, (H x W))
            x = x.reshape(b, f, (h * w))
            x = x.permute(1, 0, 2).reshape(f, (b * h * w))
            x = x.permute(1, 0)
            x = self.feature_classifier(x)
            x = x.permute(1, 0).reshape(self.num_classes, b, (h * w))
            x = x.reshape(self.num_classes, b, h, w)
            x = x.permute(1, 0, 2, 3)

        return x

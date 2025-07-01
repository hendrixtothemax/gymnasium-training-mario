import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNNWithStateExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        image_shape = observation_space["image"].shape  # (1, 144, 60)
        state_dim = observation_space["state"].shape[0]  # 279

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # compute output size of CNN automatically
        with torch.no_grad():
            dummy_input = torch.zeros(1, *image_shape)
            cnn_output_dim = self.cnn(dummy_input).shape[1]

        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(cnn_output_dim + 128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        image = observations["image"].float() / 255.0  # (batch, 1, 144, 60)
        state = observations["state"].float()  # (batch, state_dim)

        cnn_out = self.cnn(image)
        state_out = self.state_mlp(state)

        combined = torch.cat((cnn_out, state_out), dim=1)
        return self.final_mlp(combined)

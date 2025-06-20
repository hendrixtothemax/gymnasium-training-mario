from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from pyboy import PyBoy
import torch as th
import torch.nn as nn
import numpy as np
import gymnasium as gym

from mario_env import GenericPyBoyEnv

# --- Custom feature extractor that flattens and uses MLP ---
class FlatMLPFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        self.flatten = nn.Flatten()
        n_input = int(np.prod(observation_space.shape))

        self.mlp = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.mlp(self.flatten(observations))


# --- Custom policy using the MLP feature extractor ---
class FlatMLPPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=FlatMLPFeatures,
            features_extractor_kwargs=dict(features_dim=128),
        )


# --- Initialize PyBoy emulator ---
pyboy = PyBoy("../roms/SML.gb", window="null")

# --- Initialize environment ---
env = GenericPyBoyEnv(pyboy, debug=False)

# Optional: run environment checker to validate
check_env(env, warn=True)

# --- Create PPO model using custom MLP policy ---
model = PPO("MlpPolicy", env, verbose=1, device='cpu', ent_coef=0.1, n_steps=4096, batch_size=512)

# --- Train the model ---
model.learn(total_timesteps=1025000, progress_bar=True)

# --- Save the model ---
model.save("mlp_ppo_mario_500000")

# --- Close the environment after training ---
env.close()

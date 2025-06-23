import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.plugins.game_wrapper_super_mario_land import GameWrapperSuperMarioLand

class SuperMarioLandEnv(gym.Env):
    def __init__(self, rom_path="../roms/SML.gb"):
        super().__init__()

        self.pyboy = PyBoy(rom_path, window="null", sound=False)
        self.game = GameWrapperSuperMarioLand(self.pyboy, self.pyboy._plugin_manager, None)
        self.action_space = spaces.Discrete(5)  # Up to 5 discrete actions
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pyboy = PyBoy(self.pyboy.cartridge_filename, window_type="headless", sound=False)
        self.game = GameWrapperSuperMarioLand(self.pyboy)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.game.perform_action(action)
        self.pyboy.tick()
        reward = self.game.get_reward()
        done = self.game.game_over()
        obs = self._get_obs()
        return obs, reward, done, False, {}

    def _get_obs(self):
        return self.game.get_screen_image()

    def render(self):
        self.game.print_state()

    def close(self):
        self.pyboy.stop()

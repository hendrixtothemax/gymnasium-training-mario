import gymnasium as gym
import numpy as np
from pyboy import PyBoy
from gymnasium.spaces import Dict, Box, MultiBinary
from PIL import Image

actions = ['a', 'b', 'left', 'right', 'up', 'down']
# MAX_STORED_PREV_MOVES = int(len(actions) * 45)
ROM_LOCATION = "../roms/SML.gb"

class Mario(gym.Env):
    def __init__(self, window='null', frameskip=1):
        super().__init__()
        self.pyboy = PyBoy(ROM_LOCATION, window=window)
        self._fitness = 0
        self._previous_fitness = 0
        self.passed_actions = 0
        self.frame_skip = frameskip
        self.total_steps = 0
        self.passedCheckpoint = False
        self.prevpassedCheckpoint = False

        self.action_space = MultiBinary(len(actions))

        # Observation space now includes grayscale image (1, 144, 160) and state vector
        self.observation_space = Dict({
            "image": Box(low=0, high=255, shape=(1, 144, 160), dtype=np.uint8),
            "state": Box(low=-1024, high=1024, shape=(9,), dtype=np.float32)
        })

        self.pyboy.game_wrapper.start_game()

    def get_state_vector(self):
        gw = self.pyboy.game_wrapper

        x_pos = float(gw.level_progress)
        y_pos = float(self.pyboy.memory[0xC201])
        score = float(gw.score)

        small      = float(self.pyboy.memory[0xFF99] == 0x00)
        growing    = float(self.pyboy.memory[0xFF99] == 0x01)
        big        = float(self.pyboy.memory[0xFF99] == 0x02)
        shrinking  = float(self.pyboy.memory[0xFF99] == 0x03)
        invincible = float(self.pyboy.memory[0xFF99] == 0x04)
        powerup    = float(self.pyboy.memory[0xFFB5] == 0x02)

        # for i, pressed in enumerate(action):
        #     if pressed:
        #         self.prev_moves.append(float(i))

        return np.array([
            x_pos, y_pos, score,
            small, growing, big, shrinking, invincible, powerup
        ], dtype=np.float32)

    def get_image_frame(self):
        rgb_frame = self.pyboy.screen.ndarray  # (144, 160, 3)
        gray_frame = np.asarray(Image.fromarray(rgb_frame).convert("L"))  # (144, 160)
        return gray_frame[np.newaxis, :, :].astype(np.uint8)  # (1, 144, 160)

    def step(self, action):
        gw = self.pyboy.game_wrapper
        any_pressed = any(action)

        for i, pressed in enumerate(action):
            if pressed:
                self.pyboy.button(actions[i], self.frame_skip)

        self.total_steps += 1

        if not any_pressed:
            self.passed_actions += self.frame_skip
        else:
            self.passed_actions = 0

        self.prevx = gw.level_progress

        for _ in range(self.frame_skip):
            self.pyboy.tick(1, True)
            

        done = gw.game_over()
        reward = gw.score / 1000
        reward += (gw.level_progress-200) / 100
        if (self.prevx + 2) >= gw.level_progress:
            reward -= ((gw.level_progress-200) / 100) / 1.5

        if done:
            if gw.level_progress < 320:
                reward -= 210
            elif gw.level_progress < 375:
                reward -= 150
            else:
                reward += 50

        state = self.get_state_vector()
        image = self.get_image_frame()

        return {"image": image, "state": state}, reward, done, False, {}

    def reset(self, **kwargs):
        self.done = False
        self.pyboy.game_wrapper.reset_game()
        self.pyboy.game_wrapper.set_lives_left(0)
        # self.prev_moves = deque([-1.0] * MAX_STORED_PREV_MOVES, maxlen=MAX_STORED_PREV_MOVES)
        state = self.get_state_vector()
        image = self.get_image_frame()
        return {"image": image, "state": state}, {}

    def render(self, mode='human'):
        pass

    def close(self):
        self.pyboy.stop()

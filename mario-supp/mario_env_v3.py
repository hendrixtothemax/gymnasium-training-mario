import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

actions = ['a', 'b', 'left', 'right', 'up', 'down']

MAX_STORED_PREV_MOVES = int(len(actions) * 45)
GAME_OBERSVATION_SPACE = spaces.Box(low=-1024, high=1024, shape=(MAX_STORED_PREV_MOVES + 9,), dtype=np.float32)
ROM_LOCATION = "../roms/SML.gb"

class Mario(gym.Env):

    def __init__(self, window='null', frameskip=int(1)):
        super().__init__()
        from pyboy import PyBoy
        self.pyboy = PyBoy(ROM_LOCATION, window=window)
        self._fitness=0
        self._previous_fitness=0
        self.passed_actions = 0
        self.frame_skip = frameskip
        self.total_steps = 0
        self.passedCheckpoint = False
        self.prevpassedCheckpoint = False

        self.action_space = spaces.MultiBinary(len(actions))
        self.observation_space = GAME_OBERSVATION_SPACE

        self.pyboy.game_wrapper.start_game()

    def step(self, action):
        game_wrapper = self.pyboy.game_wrapper

        # Move the agent
        # Move the agent
        any_pressed = False

        for i, pressed in enumerate(action):
            if pressed:  # pressed is 1 or 0
                self.pyboy.button(actions[i], self.frame_skip)
                any_pressed = True
        
        self.total_steps += 1

        if not any_pressed:
            self.passed_actions += self.frame_skip
        else:
            self.passed_actions = 0


        prev_x = game_wrapper.level_progress

        for _ in range(self.frame_skip):
            self.pyboy.tick(1, True)

        done = self.pyboy.game_wrapper.game_over()

        if done:
            if game_wrapper.level_progress < 320:  # stuck very close to start
                self.reward -= 210  # stronger penalty
            elif game_wrapper.level_progress < 375:
                self.reward -= 150
            else:
                self.reward += 50
            self.reset()
        else:
            # self.reward = self.pyboy.game_wrapper.score / 100
            self.reward = (game_wrapper.level_progress - prev_x)
            self.reward += game_wrapper.score / 1000
            self.reward += (game_wrapper.level_progress) / 3
            self.reward += (1 / (game_wrapper.time_left + 1)) * 3

            if abs(game_wrapper.level_progress - prev_x) < 1.0:
                self.reward -= 0.1  # small penalty for no movement
            if game_wrapper.level_progress > 320:
                self.reward += 2
            if game_wrapper.level_progress > 340:
                self.reward += 5
            if game_wrapper.level_progress > 375:
                self.reward += 30
                prev_checkpoint = True
                if self.prevpassedCheckpoint != self.passedCheckpoint:
                    self.passedCheckpoint = True
                    self.reward += 2500
                    print("Passed Important Point")
            if game_wrapper.level_progress < 260 and game_wrapper.time_left < 395:
                self.reward -= 5

            if action[0] == 1 and action[3] == 1 and game_wrapper.level_progress > 300:
                self.reward += 250

            if(self.reward > 700):
                print(self.reward)

            self.reward += 1.0

            self.update_obs(action)

        info = {}
        truncated = False

        return self.observation, self.reward, done, truncated, info

    def reset(self, **kwargs):
        self.done = False
        self.pyboy.game_wrapper.reset_game()
        self.pyboy.game_wrapper.set_lives_left(0)

        # Observation: x_pos, y_pos, score, upgrade_state
        self.x_pos = float(self.pyboy.game_wrapper.level_progress)
        self.y_pos = float(self.pyboy.memory[0xC201])
        self.score = float(self.pyboy.game_wrapper.score)

        # Convert boolean flags to float (0.0 or 1.0)
        self.small      = float(self.pyboy.memory[0xFF99] == 0x00)
        self.growing    = float(self.pyboy.memory[0xFF99] == 0x01)
        self.big        = float(self.pyboy.memory[0xFF99] == 0x02)
        self.shrinking  = float(self.pyboy.memory[0xFF99] == 0x03)
        self.invincible = float(self.pyboy.memory[0xFF99] == 0x04)
        self.powerup    = float(self.pyboy.memory[0xFFB5] == 0x02)

        # Initialize prev_moves deque
        self.prev_moves = deque(maxlen=MAX_STORED_PREV_MOVES)
        for _ in range(MAX_STORED_PREV_MOVES):
            self.prev_moves.append(-1.0)

        # Final observation as a float32-compatible list
        self.observation = [
            self.x_pos,
            self.y_pos,
            self.score,
            self.small,
            self.growing,
            self.big,
            self.shrinking,
            self.invincible,
            self.powerup
        ] + list(self.prev_moves)

        self.observation = np.array(self.observation, dtype=np.float32)

        info = {}
        return self.observation, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.pyboy.stop()

    def update_obs(self, action: float):
        # Observation: x_pos, y_pos, score, upgrade_state
        self.x_pos = float(self.pyboy.game_wrapper.level_progress)
        self.y_pos = float(self.pyboy.memory[0xC201])
        self.score = float(self.pyboy.game_wrapper.score)

        # Convert boolean flags to float (0.0 or 1.0)
        self.small      = float(self.pyboy.memory[0xFF99] == 0x00)
        self.growing    = float(self.pyboy.memory[0xFF99] == 0x01)
        self.big        = float(self.pyboy.memory[0xFF99] == 0x02)
        self.shrinking  = float(self.pyboy.memory[0xFF99] == 0x03)
        self.invincible = float(self.pyboy.memory[0xFF99] == 0x04)
        self.powerup    = float(self.pyboy.memory[0xFFB5] == 0x02)

        for i, pressed in enumerate(action):
            self.prev_moves.append(float(i))

        # Final observation as a float32-compatible list
        self.observation = [
            self.x_pos,
            self.y_pos,
            self.score,
            self.small,
            self.growing,
            self.big,
            self.shrinking,
            self.invincible,
            self.powerup
        ] + list(self.prev_moves)

        self.observation = np.array(self.observation, dtype=np.float32)

# Adopted from https://github.com/NicoleFaye/PyBoy/blob/rl-test/PokemonPinballEnv.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy

actions = ['','a', 'b', 'left', 'right', 'up', 'down']

matrix_shape = (320,)
game_area_observation_space = spaces.Box(low=0, high=371, shape=matrix_shape, dtype=np.uint32)

class GenericPyBoyEnv(gym.Env):

    def __init__(self, pyboy: PyBoy, debug=False, frame_skip=15, show=False):
        super().__init__()
        self.pyboy = pyboy
        self._fitness=0
        self._previous_fitness=0
        self.debug = debug
        self.passed_actions = 0
        self.frame_skip = frame_skip
        self.show = show

        if not self.debug:
            self.pyboy.set_emulation_speed(0)

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = game_area_observation_space

        self.pyboy.game_wrapper.start_game()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        game_wrapper = self.pyboy.game_wrapper

        # Move the agent
        if action == 0:
            self.passed_actions += self.frame_skip
        else:
            self.pyboy.button(actions[action], self.frame_skip-1)
            self.passed_actions = 0

        # Consider disabling renderer when not needed to improve speed:
        # self.pyboy.tick(1, False)
        prevLoc = game_wrapper.level_progress
        prevLives = game_wrapper.lives_left
        prevY = self.pyboy.memory[0xC201]

        for _ in range(self.frame_skip):
            self.pyboy.tick(1, True)  

        curY = self.pyboy.memory[0xC201]
        yChange = prevY - curY

        curLoc = game_wrapper.level_progress
        curSpeed = curLoc - prevLoc

        curLives = game_wrapper.lives_left
        died = (curLives < prevLives)

        done = self.pyboy.game_wrapper.game_over()

        args = {"action": action, "speed": curSpeed, "died": died, "y": yChange, "cur_y": curY, "prevX": prevLoc, "x": curLoc}

        self._calculate_fitness(args)
        reward=self._fitness-self._previous_fitness

        if done:
            self.reset()

        observation=self.pyboy.game_area()
        observation = np.array(observation).flatten()
        info = {}
        truncated = False

        return observation, reward, done, truncated, info

    def _calculate_fitness(self, args):
        self._previous_fitness = self._fitness
        now_score = 0

        game_wrapper = self.pyboy.game_wrapper

        if self.passed_actions > self.frame_skip * 10:
            now_score -= (self.passed_actions - (self.frame_skip * 10)) // self.frame_skip

        if args["speed"] > 0:
            now_score += 5 * args["speed"]
        if args["died"]:
            now_score -= 100  # Punish dying
        if args["prevX"] < args["x"]:
            now_score += 2

        self._fitness = now_score


    def reset(self, **kwargs):
        self.pyboy.game_wrapper.reset_game()
        self._fitness=0
        self._previous_fitness=0

        self.pyboy.game_wrapper.set_lives_left(0)

        observation=self.pyboy.game_area()
        observation = np.array(observation).flatten()
        info = {}
        return observation, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.pyboy.stop()

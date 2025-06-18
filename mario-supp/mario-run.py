from pyboy import PyBoy
from mario_env import GenericPyBoyEnv

# pyboy = PyBoy("../roms/SML.gb", window="SDL2", scale=6)
pyboy = PyBoy("../roms/SML.gb")

marioEnv = GenericPyBoyEnv(pyboy)
from pyboy import PyBoy
import time

# https://docs.pyboy.dk/index.html

pyboy = PyBoy("./roms/SMBD.gbc", window="SDL2", scale=6)

while True:
    pyboy.tick()
    time.sleep(0.01)  # Slow down so you can see it

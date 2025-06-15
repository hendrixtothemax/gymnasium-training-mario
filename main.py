from pyboy import PyBoy
from pyboy.windowevent import WindowEvent

import time

# https://docs.pyboy.dk/index.html

SAVE_STATE_SLOT = 1  # slot to save to
pyboy = PyBoy("./roms/SMBD.gbc", window="SDL2", scale=6)

pyboy.set_emulation_speed(0)  # Full speed

# Let the game initialize
for _ in range(100):
    pyboy.tick()

    running = True
    while running:
        # Handle user input
        if pyboy.get_input():
            if pyboy.window.get_pressed_keys():
                keys = pyboy.window.get_pressed_keys()
                if WindowEvent.PRESS_ARROW_LEFT in keys:
                    print("Left arrow pressed")

            # Check for custom hotkey (e.g. F5 for save state)
            if pyboy.window.get_event(WindowEvent.PRESS_BUTTON_START):
                print("Saving state at Mario's starting position...")
                pyboy.save_state(SAVE_STATE_SLOT)

            # Press ESC to quit
            if pyboy.window.get_event(WindowEvent.PRESS_BUTTON_SELECT):
                print("Exiting...")
                running = False

        pyboy.tick()

    time.sleep(0.01)  # Slow down so you can see it

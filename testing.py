from pyboy import PyBoy
# import keyboard

pyboy = PyBoy("./roms/SMBD.gbc", window="SDL2", scale=6)

# with open('roms/SMBD_save.state', 'rb') as f:
#    pyboy.load_state(f)

def save_state_to_file(something):
    with open('roms/SMBD_save.state', 'wb') as f:
        pyboy.save_state(f)


# keyboard.on_press_key('m', save_state_to_file)

cur_frame = 0
while pyboy.tick():

    cur_frame += 1
    if cur_frame % 15 == 0:
        x_low, x_high = pyboy.memory[0xC1CA:0xC1CC]
        y_low, y_high = pyboy.memory[0xC1CC:0xC1CE]

        mario_x = x_low + (x_high << 8)
        mario_y = y_low + (y_high << 8)


        print(f"Frame: {cur_frame} | Mario POS: ({mario_x}, {mario_y})", end="\r", flush=True)
    
    if cur_frame >= 60:
        cur_frame = 0

pyboy.stop()

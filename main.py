from pyboy import PyBoy

pyboy = PyBoy("./roms/SMBD.gbc", window="SDL2", scale=6)

def dump_memory_hex(memory, start_addr, end_addr):
    for addr in range(start_addr, end_addr + 1):
        byte = memory[addr]
        print(f"0x{addr:04X}: 0x{byte:02X}")


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

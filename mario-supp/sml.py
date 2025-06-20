from pyboy import PyBoy
# import keyboard

pyboy = PyBoy("../roms/SML.gb", window="SDL2", scale=6)
pyboy.set_emulation_speed(1)
pyboy.cartridge_title == "SUPER MARIOLAN"

mario = pyboy.game_wrapper
mario.game_area_mapping(mario.mapping_compressed, 0)
mario.start_game()

tick = 0
while pyboy.tick():
    if tick % 30 == 0:
        gamearea = pyboy.game_area()
        print(gamearea)
        print(f'Height: {len(gamearea)} x Length: {len(gamearea[0])}')
        tick = 0
    tick += 1

pyboy.stop()
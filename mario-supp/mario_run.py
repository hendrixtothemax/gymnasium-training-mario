from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from pyboy import PyBoy

from mario_env import GenericPyBoyEnv
from matio_env_v2 import Mario

# --- Initialize PyBoy emulator ---
pyboy = PyBoy("../roms/SML.gb", window="null")

# --- Initialize environment ---
env = Mario(pyboy, frameskip=2)

# Optional: run environment checker to validate
check_env(env, warn=True)

# --- Create PPO model using custom MLP policy ---
model = PPO("MlpPolicy", env, verbose=1, device='cpu', ent_coef=0.35, n_steps=512, batch_size=256, tensorboard_log="./tensorboard_logs/")

# --- Create checkpoint callback to save every 25,000 steps ---
checkpoint_callback = CheckpointCallback(
    save_freq=100000,  # Number of steps
    save_path="./checkpoints/",
    name_prefix="ppo_mario",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

# --- Train the model ---
model.learn(total_timesteps=5000000, progress_bar=True, callback=checkpoint_callback, tb_log_name="ppo_mario_run")

# --- Save the model ---
model.save("mlp_ppo_mario_final")

# --- Close the environment after training ---
env.close()

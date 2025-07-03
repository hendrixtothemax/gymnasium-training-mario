from stable_baselines3 import PPO
from pyboy import PyBoy
from mario_env import GenericPyBoyEnv
from mario_env_v4 import Mario

# Load PPO model
model = PPO.load("/home/tsubaki/projects/gymnasium-training-mario/mario-supp/checkpointsv4/mario_ppo_cnn_1mil_800000_steps.zip", device='cpu')

# Set up emulator and environment
env = Mario(window='SDL2')

obs, info = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3 import PPO  # Or your chosen algorithm

# 1. Load the trained model
model = PPO.load("../ppo-LunarLander-v3.zip")  # Replace with your model file

# 2. Create and wrap the environment
env = gym.make("LunarLander-v3", render_mode="rgb_array") #Specify render mode to enable video recording
env = RecordEpisodeStatistics(env)
env = RecordVideo(env, "videos", step_trigger=lambda t: t % 500 == 0) # Record a video every 100 steps.

# 3. Run the agent
observation, info = env.reset()
for _ in range(5000):  # Run for a certain number of steps or episodes
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
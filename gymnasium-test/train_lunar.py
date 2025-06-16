import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# First, we create our environment called LunarLander-v2
env = gym.make("LunarLander-v3")

# Then we reset this environment
observation, info = env.reset()

print("_____OBSERVATION SPACE_____ \n")
print("Observation Space Shape", env.observation_space.shape)
print("Sample observation", env.observation_space.sample())

# Create the environment
env = make_vec_env("LunarLander-v3", n_envs=16)

# Instantiate the agent
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1,
    device='cpu',
)

# SOLUTION
# Train it for 1,000,000 timesteps
model.learn(total_timesteps=1000000)
# Save the model
model_name = "ppo-LunarLander-v3"
model.save(model_name)

# for _ in range(20):
#     # Take a random action
#     action = env.action_space.sample()
#     print("Action taken:", action)

#     # Do this action in the environment and get
#     # next_state, reward, terminated, truncated and info
#     observation, reward, terminated, truncated, info = env.step(action)

#     # If the game is terminated (in our case we land, crashed) or truncated (timeout)
#     if terminated or truncated:
#         # Reset the environment
#         print("Environment is reset")
#         observation, info = env.reset()

# env.close()
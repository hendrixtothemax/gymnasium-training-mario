from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from smland_env import SuperMarioLandEnv

def make_env():
    return SuperMarioLandEnv()

if __name__ == "__main__":
    env = DummyVecEnv([make_env])

    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)

    model.save("ppo_mario_land")

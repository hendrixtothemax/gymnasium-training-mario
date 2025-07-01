from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
from pyboy import PyBoy

from mario_env_v3 import Mario

NUM_ENVS = 16

def make_env():
    def _init():
        return Mario(window='null')
    return _init

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("fork")  # Important for PyBoy + multiprocessing

    # --- Initialize vectorized environment ---
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    # Optional: validate one of the environments
    check_env(make_env()(), warn=True)

    # --- Create PPO model using custom MLP policy ---
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device='cpu',  # Use 'auto' instead of 'gpu' unless you know the exact device
        ent_coef=0.35,
        n_steps=60,  # Not multiplied — SB3 handles vectorization
        batch_size=64,  # Also not multiplied
        tensorboard_log="./tensorboard_logs/"
    )

    # --- Create checkpoint callback to save every 100,000 steps ---
    checkpoint_callback = CheckpointCallback(
        save_freq=500000,
        save_path="./checkpoints/",
        name_prefix="ppo_mario_large",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # --- Train the model ---
    model.learn(
        total_timesteps=50_000_000,
        progress_bar=True,
        callback=checkpoint_callback,
        tb_log_name="ppo_mario_run_large"
    )

    # --- Save the final model ---
    model.save("mlp_ppo_mario_50_000_000_final")

    # --- Close the environment ---
    env.close()

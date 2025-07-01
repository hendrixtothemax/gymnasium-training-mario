from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from cnn_with_state_extractor import CNNWithStateExtractor
from mario_env_v4 import Mario

NUM_ENVS = 32

def make_env():
    def _init():
        return Mario(window='null', frameskip=2)
    return _init

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("fork")  # Important for PyBoy + multiprocessing

    # --- Initialize vectorized environment ---
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    # Optional: validate one of the environments
    check_env(make_env()(), warn=True)


    policy_kwargs = dict(
        features_extractor_class=CNNWithStateExtractor,
        features_extractor_kwargs=dict(features_dim=256)
    )

    # --- Create PPO model using custom MLP policy ---
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        device='auto',  # Use 'auto' instead of 'gpu' unless you know the exact device
        ent_coef=0.35,
        n_steps=2048,  # Not multiplied — SB3 handles vectorization
        batch_size=512,  # Also not multiplied
        tensorboard_log="./tensorboard_logs/"
    )

    # --- Create checkpoint callback to save every 100,000 steps ---
    checkpoint_callback = CheckpointCallback(
        save_freq=131072,
        save_path="./checkpointsv4/",
        name_prefix="mario_ppo_cnn_5mil",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # --- Train the model ---
    model.learn(
        total_timesteps=5_000_000,
        progress_bar=True,
        callback=checkpoint_callback,
        tb_log_name="mario_ppo_cnn_5mil"
    )

    # --- Save the final model ---
    model.save("mario_ppo_cnn_v4_5mil")

    # --- Close the environment ---
    env.close()

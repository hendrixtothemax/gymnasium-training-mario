from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from cnn_with_state_extractor import CNNWithStateExtractor
from stable_baselines3.common.monitor import Monitor
from mario_env_v4 import Mario

NUM_ENVS = 16

def make_env():
    def _init():
        return Monitor(Mario(window='null', frameskip=15))
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
        device='cuda',  # Use 'auto' instead of 'gpu' unless you know the exact device
        ent_coef=0.35,
        n_steps=120,  # Not multiplied — SB3 handles vectorization
        batch_size=64,  # Also not multiplied
        tensorboard_log="./tensorboard_logs/"
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/eval/",
        eval_freq=8000,  # Evaluate every 50k steps
        deterministic=True,
        render=False
    )

    # # --- Create checkpoint callback to save every 100,000 steps ---
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpointsv4/",
        name_prefix="mario_ppo_cnn_1mil",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )


    callback = CallbackList([checkpoint_callback, eval_callback])

    # --- Train the model ---
    model.learn(
        total_timesteps=1_000_000,
        progress_bar=True,
        callback=callback,
        tb_log_name="mario_ppo_cnn_1mil"
    )

    # --- Save the final model ---
    model.save("mario_ppo_cnn_v4_1mil")

    # --- Close the environment ---
    env.close()

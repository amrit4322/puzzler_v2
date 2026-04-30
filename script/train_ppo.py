import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from env.brainiton_env_v2 import BrainItOnGeneralEnv


MODEL_DIR = "models"
LOG_DIR = "logs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def make_env(level_path, reward_mode="dense"):
    def _init():
        env = BrainItOnGeneralEnv(
            level_path=level_path,
            render_mode=None,
            reward_mode=reward_mode,
            control_mode="agent",
            max_steps=80,
        )
        return Monitor(env)

    return _init


def train():
    level_path="levels/level_boal_goal.json"

    train_env = DummyVecEnv([
        make_env(level_path, reward_mode="dense")
    ])
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv([
        make_env(level_path, reward_mode="dense")
    ])
    eval_env = VecMonitor(eval_env)

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=MODEL_DIR,
        name_prefix="ppo_general_checkpoint",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=5_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="auto",
    )

    model.learn(
        total_timesteps=200_000,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    model.save(f"{MODEL_DIR}/ppo_general_level1_dense_final")
    print("Model saved.")


if __name__ == "__main__":
    train()
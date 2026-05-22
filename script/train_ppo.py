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


def make_env(
    level_path,
    level_id=1,
    reward_mode="dense",
    stroke_body="static",
    max_steps=500,
    agent_draw_mode="stroke",
    num_stroke_points=2,
):
    def _init():
        env = BrainItOnGeneralEnv(
            level_path=level_path,
            level_id=level_id,
            render_mode=None,
            reward_mode=reward_mode,
            control_mode="agent",
            max_steps=max_steps,
            stroke_body=stroke_body,
            agent_draw_mode=agent_draw_mode,
            num_stroke_points=num_stroke_points,
        )
        return Monitor(env,info_keywords=("is_success", "is_failure"))

    return _init


def train():
    level_path = "levels/level.json"
    level_id = 1
    reward_mode = "dense"
    stroke_body = "static"
    max_steps = 100
    model_name = f"ppo_level{level_id}_{reward_mode}_{stroke_body}"

    agent_draw_mode = "stroke"
    num_stroke_points = 2
    train_env = DummyVecEnv([
        make_env(
            level_path=level_path,
            level_id=level_id,
            reward_mode=reward_mode,
            stroke_body=stroke_body,
            max_steps=max_steps,
            agent_draw_mode=agent_draw_mode,
num_stroke_points=num_stroke_points,
        )
    ])
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv([
        make_env(
            level_path=level_path,
            level_id=level_id,
            reward_mode=reward_mode,
            stroke_body=stroke_body,
            max_steps=max_steps,
            agent_draw_mode=agent_draw_mode,
num_stroke_points=num_stroke_points,
        )
    ])
    eval_env = VecMonitor(eval_env)

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=MODEL_DIR,
        name_prefix=f"{model_name}_checkpoint",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=5_000,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=1024,
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
        total_timesteps=30_000,
        callback=[checkpoint_callback],
        progress_bar=True,
        tb_log_name=model_name,
    )

    final_model_path = f"{MODEL_DIR}/{model_name}_final"
    model.save(final_model_path)

    train_env.close()
    eval_env.close()

    print(f"Model saved at: {final_model_path}")


if __name__ == "__main__":
    train()
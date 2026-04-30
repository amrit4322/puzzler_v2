import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
from stable_baselines3 import PPO

from env.brainiton_env_v2 import BrainItOnGeneralEnv


def evaluate(render=True, episodes=5):
    model_path = "models/ppo_general_level1_dense_final"
    level_path="levels/level_boal_goal.json"

    model = PPO.load(model_path)

    env = BrainItOnGeneralEnv(
        level_path=level_path,
        render_mode="human" if render else None,
        reward_mode="dense",
        control_mode="agent",
        max_steps=80,
    )

    for ep in range(episodes):
        obs, info = env.reset()

        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        print(
            f"Episode {ep + 1}: "
            f"reward={total_reward:.2f}, "
            f"success={info['is_success']}, "
            f"steps={info['step_count']}, "
            f"segments={info['segments_used']}, "
            f"distance={info['goal_distance']:.2f}"
        )

    env.close()


if __name__ == "__main__":
    evaluate(render=True, episodes=5)
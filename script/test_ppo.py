import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from stable_baselines3 import PPO

from env.brainiton_env_v2 import BrainItOnGeneralEnv


def evaluate(render=True, episodes=5):
    level_path = "levels/level.json"
    level_id = 1
    reward_mode = "dense"
    stroke_body = "static"
    max_steps = 100
    agent_draw_mode = "stroke"
    num_stroke_points = 2
    render_mode = "human"

    model_path = f"models/ppo_level{level_id}_{reward_mode}_{stroke_body}_final"

    model = PPO.load(model_path)

    env = BrainItOnGeneralEnv(
         level_path=level_path,
        level_id=level_id,
        render_mode="human" if render else None,
        reward_mode=reward_mode,
        control_mode="agent",
        max_steps=max_steps,
        stroke_body=stroke_body,
        agent_draw_mode=agent_draw_mode,
        num_stroke_points=num_stroke_points,
    )

    success_count = 0
    total_rewards = []
    total_steps = []
    final_distances = []

    for ep in range(episodes):
        obs, info = env.reset()

        terminated = False
        truncated = False
        episode_reward = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            print("Action:", action)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
    

            if render:
                env.render()
                time.sleep(0.03)

        success = info.get("is_success", False)
        steps = info.get("step_count", max_steps)
        distance = info.get("goal_distance", None)
        segments = info.get("segments_used", None)

        if success:
            success_count += 1

        total_rewards.append(episode_reward)
        total_steps.append(steps)

        if distance is not None:
            final_distances.append(distance)

        print(
            f"Episode {ep + 1}: "
            f"reward={episode_reward:.2f}, "
            f"success={success}, "
            f"steps={steps}, "
            f"segments={segments}, "
            f"distance={distance:.2f}" if distance is not None else ""
        )

    print("\nEvaluation Summary")
    print("------------------")
    print(f"Episodes: {episodes}")
    print(f"Success Rate: {(success_count / episodes) * 100:.2f}%")
    print(f"Average Reward: {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"Average Steps: {sum(total_steps) / len(total_steps):.2f}")

    if final_distances:
        print(f"Average Final Distance: {sum(final_distances) / len(final_distances):.2f}")

    env.close()


if __name__ == "__main__":
    evaluate(render=True, episodes=5)
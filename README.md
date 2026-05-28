# puzzler_v2

Brain-It-On style physics puzzle environment for human play, PPO training, PPO evaluation, and LLM-based stroke testing.

The main files are:

- `env/brainiton_env_v2.py` - Gymnasium environment.
- `levels/level.json` - all level definitions.
- `script/run.py` - run levels manually or with random agent actions.
- `script/train_ppo.py` - train PPO.
- `script/test_ppo.py` - evaluate and visualize PPO.
- `script/test_llm.py` - ask an LLM for stroke points and visualize the result.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the packages used by the project:

```bash
pip install gymnasium numpy pygame pymunk stable-baselines3 tensorboard openai
```

If you use the LLM test script, set your OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Check that the key is available:

```bash
echo ${OPENAI_API_KEY:0:10}
```

## Run Human Mode

Run level 1 in human mode:

```bash
python script/run.py --mode human --level 1
```

Run a different level:

```bash
python script/run.py --mode human --level 10
```

Use static strokes:

```bash
python script/run.py --mode human --level 10 --stroke_body static
```

Use dynamic strokes:

```bash
python script/run.py --mode human --level 10 --stroke_body dynamic
```

Useful controls:

```text
Drag mouse = draw stroke
R = reset level
C = clear drawn strokes
N = next level
P = previous level
Space = pause / unpause simulation
Esc = quit
```

## Train PPO

Train PPO with the default multi-level setup:

```bash
python script/train_ppo.py
```

By default this trains on:

```text
1, 2, 5, 7, 8, 9, 10, 11, 12, 14, 16, 17
```

And evaluates during training on:

```text
3, 4, 6, 13, 15, 18, 19, 20
```

Train on specific levels:

```bash
python script/train_ppo.py --train-levels 1,2,5,7,8 --eval-levels 3,4,6
```

Train with dense reward:

```bash
python script/train_ppo.py --reward-mode dense
```

Train with sparse reward:

```bash
python script/train_ppo.py --reward-mode sparse
```



Train with stroke mode using 2 points:

```bash
python script/train_ppo.py --agent-draw-mode stroke --num-stroke-points 2
```

Train with stroke mode using 3 points:

```bash
python script/train_ppo.py --agent-draw-mode stroke --num-stroke-points 3
```

Train for more timesteps:

```bash
python script/train_ppo.py --total-timesteps 1000000
```

Train faster with fewer parallel environments:

```bash
python script/train_ppo.py --n-envs 2
```

The trained model is saved under:

```text
models/ppo_multilevel_dense_static_stroke_2pt/
```

Important saved files:

```text
best_model.zip
final_model.zip
```

The script does not save checkpoint files at every timestep, so the `models` folder should not fill up quickly.

## Test PPO

Test the default model on the default evaluation levels:

```bash
python script/test_ppo.py
```

Test and visualize PPO:

```bash
python script/test_ppo.py --render
```

Test specific levels:

```bash
python script/test_ppo.py --levels 1,2,3,4,5
```

Test one episode per level:

```bash
python script/test_ppo.py --levels 1,2,3 --episodes-per-level 1
```

Pause after each episode so you can inspect the final state:

```bash
python script/test_ppo.py --levels 1,2,3 --episodes-per-level 1 --render --pause-after-episode 3
```

Print the stroke coordinates drawn by PPO:

```bash
python script/test_ppo.py --levels 1,2,3 --episodes-per-level 1 --render --print-stroke
```

Use stochastic evaluation to see different attempts:

```bash
python script/test_ppo.py --levels 2 --episodes-per-level 5 --render --stochastic --print-stroke
```

Test a specific model path:

```bash
python script/test_ppo.py --model-path models/ppo_multilevel_dense_static_stroke_2pt/best_model.zip --levels 1,2,3 --render
```

If you trained with 3 stroke points, test with 3 stroke points:

```bash
python script/test_ppo.py --agent-draw-mode stroke --num-stroke-points 3 --render
```

Training and testing must use the same action setup:

```text
stroke_body
agent_draw_mode
num_stroke_points
reward_mode
```

## Test With LLM

The LLM test script sends a level JSON to an LLM, asks it to return stroke points, draws the stroke, and runs the simulation.

Run one LLM attempt on level 1:

```bash
python script/test_llm.py --levels 1 --render
```

Run multiple levels:

```bash
python script/test_llm.py --levels 1,2,3,4 --attempts-per-level 1 --render
```

Run multiple attempts on one level:

```bash
python script/test_llm.py --levels 2 --attempts-per-level 5 --temperature 0.7 --render
```

Use 3-point strokes:

```bash
python script/test_llm.py --levels 1,2,3 --num-stroke-points 3 --render
```

Change the LLM model:

```bash
python script/test_llm.py --levels 1 --render --model gpt-5.2
```

Cheaper model option:

```bash
python script/test_llm.py --levels 1 --render --model gpt-5-mini
```

Recommended LLM models:

```text
Best quality: gpt-5.2
Good quality: gpt-5.1
Cheaper experiments: gpt-5-mini
```

## TensorBoard

Start TensorBoard:

```bash
tensorboard --logdir logs
```

Then open the URL printed by TensorBoard, usually:

```text
http://localhost:6006
```

Useful metrics:

```text
rollout/ep_rew_mean
rollout/ep_len_mean
eval/mean_reward
eval/success_rate
```

## What The PPO Test Output Means

Example:

```text
Level  1 | Episode  1 | reward=105.01 | success=True | steps=41 | segments=1 | distance=43.98 | failure=None
```

Meaning:

```text
Level = level id
Episode = attempt number
reward = total reward from the episode
success = whether the goal was solved
steps = physics steps used before success or timeout
segments = number of drawn strokes
distance = distance to the first goal center
failure = timeout, invalid_draw, or None
```

If you see:

```text
success=False | steps=500 | failure=timeout
```

The model drew a stroke, but the goal was not reached within 500 steps.

If you see the same output repeated for each episode, that is normal for deterministic testing. Use `--stochastic` to see variation.

## Common Notes

For PPO, `num_stroke_points=2` means the stroke is a straight line from point 1 to point 2.

For more flexible strokes, use:

```bash
--num-stroke-points 3
```

But this makes learning harder because the action space becomes larger.

The action size is:

```text
2 points = 4 action values
3 points = 6 action values
4 points = 8 action values
```

If macOS prints duplicate SDL warnings from `cv2` and `pygame`, it is usually a library warning, not the reason PPO fails to learn.

## Cleanup Old Checkpoints

If old checkpoint files are taking space, remove them manually:

```bash
rm models/*checkpoint*_steps.zip
```

Keep these:

```text
best_model.zip
final_model.zip
```

## Quick Command List

```bash
# Human mode
python script/run.py --mode human --level 1

# Train PPO
python script/train_ppo.py

# Test PPO without rendering
python script/test_ppo.py --levels 1,2,3

# Test PPO with rendering and stroke print
python script/test_ppo.py --levels 1,2,3 --render --print-stroke --pause-after-episode 3

# Test LLM with rendering
python script/test_llm.py --levels 1 --render

# TensorBoard
tensorboard --logdir logs
```

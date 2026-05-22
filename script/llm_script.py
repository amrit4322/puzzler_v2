import os
import json
from typing import List, Optional
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_json(level_path: str) -> dict:
    with open(level_path, "r") as f:
        return json.load(f)


def get_level_by_id(data: dict, level_id: int) -> dict:
    for level in data.get("levels", []):
        if level.get("level_id") == level_id:
            return level

    raise ValueError(f"Level ID {level_id} not found.")


def build_prompt(level_data: dict) -> str:
    return f"""
You are solving a Brain-It-On style physics puzzle.

Your task is to generate stroke points that can solve the selected level.

Return ONLY valid JSON in this exact format:

{{
  "stroke_points": [
    [x1, y1],
    [x2, y2],
    [x3, y3]
  ],
  "reason": "short explanation"
}}

Important rules:
- Use the given objects, goals, gravity, static segments, joints, and draw_rules.
- All points must be inside the draw_region.
- Do not exceed the max_segments limit.
- A stroke with N points creates N-1 line segments.
- Do not return markdown.
- Do not return text outside JSON.

Selected level:
{json.dumps(level_data, indent=2)}
"""


def ask_llm_for_stroke(level_data: dict, model: str = "gpt-4.1") -> dict:
    prompt = build_prompt(level_data)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a physics puzzle solver. Return only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    raw_output = response.choices[0].message.content

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        raise ValueError(f"LLM returned invalid JSON:\n{raw_output}")


def validate_stroke(level_data: dict, stroke_points: List[List[float]]) -> bool:
    draw_rules = level_data.get("draw_rules", {})
    region = draw_rules.get("draw_region", {})

    x_min = region.get("x_min", 0)
    x_max = region.get("x_max", 900)
    y_min = region.get("y_min", 0)
    y_max = region.get("y_max", 620)

    max_segments = draw_rules.get("max_segments", 1)

    if len(stroke_points) < 2:
        print("Invalid: stroke must contain at least 2 points.")
        return False

    segment_count = len(stroke_points) - 1

    if segment_count > max_segments:
        print(f"Invalid: stroke has {segment_count} segments, but max allowed is {max_segments}.")
        return False

    for point in stroke_points:
        if not isinstance(point, list) or len(point) != 2:
            print("Invalid point:", point)
            return False

        x, y = point

        if not (x_min <= x <= x_max):
            print(f"Invalid: x={x} is outside draw region [{x_min}, {x_max}]")
            return False

        if not (y_min <= y <= y_max):
            print(f"Invalid: y={y} is outside draw region [{y_min}, {y_max}]")
            return False

    return True


def solve_level_with_llm(
    level_path: str,
    level_id: int,
    model: str = "gpt-4.1"
) -> Optional[List[List[float]]]:

    all_data = load_json(level_path)
    level_data = get_level_by_id(all_data, level_id)

    print(f"\nSolving Level {level_id}: {level_data.get('title')}")

    result = ask_llm_for_stroke(level_data, model=model)

    stroke_points = result.get("stroke_points", [])
    reason = result.get("reason", "")

    print("\nLLM Reason:")
    print(reason)

    print("\nGenerated Stroke Points:")
    print(stroke_points)

    if validate_stroke(level_data, stroke_points):
        print("\nStroke is valid.")
        return stroke_points

    print("\nStroke is invalid.")
    return None


if __name__ == "__main__":
    level_path = "levels/level.json"
    level_id = 5

    stroke = solve_level_with_llm(
        level_path=level_path,
        level_id=level_id,
        model="gpt-4.1"
    )

    if stroke:
        print("\nReady to send to environment:")
        print(stroke)
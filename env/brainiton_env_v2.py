import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
import pymunk
from gymnasium import spaces

from env.object_factory import create_object,create_segment
from env.goal_evaluator import GoalEvaluator


class BrainItOnGeneralEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    FPS = 60
    DT = 1.0 / FPS
    MAX_SPEED = 1200.0

    def __init__(
        self,
        level_path: str,
        render_mode: Optional[str] = None,
        reward_mode: str = "dense",
        control_mode: str = "agent",
        max_steps: int = 120,
        level_id: int = 1,
        agent_draw_mode: str = "stroke",  
        stroke_body: str = "static",
        simulation_started: bool = False,
        num_stroke_points: int = 3,
    ):
        super().__init__()
        self.simulation_started = simulation_started
        self.stroke_body_type = stroke_body

        self.level_path = Path(level_path)
        self.render_mode = render_mode
        self.reward_mode = reward_mode
        self.control_mode = control_mode
        self.agent_draw_mode = agent_draw_mode
        self.max_steps = max_steps
        self.level_id = int(level_id)
        self.num_stroke_points = num_stroke_points
        
        self.all_levels = self._load_levels_file(self.level_path)
        self.level_data = self._get_level_by_id(self.level_id)

        canvas = self.level_data.get("canvas", {})
        self.canvas_width = int(canvas.get("width", 900))
        self.canvas_height = int(canvas.get("height", 620))
        self.ui_height = 130
        self.window_width = self.canvas_width
        self.window_height = self.canvas_height + self.ui_height

        draw_rules = self.level_data.get("draw_rules", {})
        self.max_drawn_segments = int(draw_rules.get("max_segments", 5))
        self.agent_segment_thickness = float(draw_rules.get("segment_thickness", 6.0))
        self.draw_region = draw_rules.get(
            "draw_region",
            {
                "x_min": 100.0,
                "x_max": 700.0,
                "y_min": 80.0,
                "y_max": 500.0,
            },
        )

        self.space = None
        self.objects: Dict[str, Dict[str, Any]] = {}
        self.static_segments = []
        self.agent_segments = []

        self.goal_evaluator = None
        self.goal_reached = False
        self.step_count = 0
        self.prev_goal_distance = 0.0

        self.screen = None
        self.clock = None
        self.font = None
        self.current_stroke_start = None
        self.current_stroke_points = []
        self.preview_stroke_points = []

        self.last_action = None
        self.last_line_length = 0.0

        self.action_space = spaces.Box(
            low=np.array(
                [
                    self.draw_region["x_min"],
                    self.draw_region["y_min"],
                    self.draw_region["x_min"],
                    self.draw_region["y_min"],
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    self.draw_region["x_max"],
                    self.draw_region["y_max"],
                    self.draw_region["x_max"],
                    self.draw_region["y_max"],
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # Generic observation for now:
        # first dynamic object position/velocity + goal center if available
        self.observation_space = spaces.Box(
            low=np.array([-1.0] * 10, dtype=np.float32),
            high=np.array([1.0] * 10, dtype=np.float32),
            dtype=np.float32,
        )

    def _load_level(self, level_path: Path) -> Dict[str, Any]:
        if not level_path.exists():
            raise FileNotFoundError(f"Level file not found: {level_path}")

        with open(level_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _load_levels_file(self, level_path: Path) -> Dict[str, Any]:
        if not level_path.exists():
            raise FileNotFoundError(f"Levels file not found: {level_path}")

        with open(level_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "levels" not in data:
            raise ValueError("levels.json must contain a top-level 'levels' list.")

        return data
    
    def _get_level_by_id(self, level_id: int) -> Dict[str, Any]:
        for level in self.all_levels["levels"]:
            if int(level["level_id"]) == int(level_id):
                return level

        available = [level["level_id"] for level in self.all_levels["levels"]]
        raise ValueError(f"Level {level_id} not found. Available levels: {available}")
    
    def get_available_level_ids(self):
        return sorted([int(level["level_id"]) for level in self.all_levels["levels"]])
    
    def load_level_by_id(self, level_id: int):
        self.level_id = int(level_id)
        self.level_data = self._get_level_by_id(self.level_id)

        canvas = self.level_data.get("canvas", {})
        self.canvas_width = int(canvas.get("width", 900))
        self.canvas_height = int(canvas.get("height", 620))

        draw_rules = self.level_data.get("draw_rules", {})
        self.max_drawn_segments = int(draw_rules.get("max_segments", 5))
        self.agent_segment_thickness = float(draw_rules.get("segment_thickness", 6.0))
        self.draw_region = draw_rules.get(
            "draw_region",
            {
                "x_min": 100.0,
                "x_max": 700.0,
                "y_min": 80.0,
                "y_max": 500.0,
            },
        )

        self.reset()


    def next_level(self):
        level_ids = self.get_available_level_ids()
        current_index = level_ids.index(self.level_id)
        next_index = (current_index + 1) % len(level_ids)
        self.load_level_by_id(level_ids[next_index])


    def previous_level(self):
        level_ids = self.get_available_level_ids()
        current_index = level_ids.index(self.level_id)
        previous_index = (current_index - 1) % len(level_ids)
        self.load_level_by_id(level_ids[previous_index])

    def _setup_world(self):
        self.space = pymunk.Space()


        physics = self.level_data.get("physics", {})

        gravity = float(physics.get("gravity", 500.0))
        damping = float(physics.get("damping", 0.98))
        iterations = int(physics.get("iterations", 20))

        self.space.gravity = (0.0, gravity)
        self.space.damping = damping
        self.space.iterations = iterations

        self.objects = {}
        self.static_segments = []
        self.agent_segments = []

        for obj_config in self.level_data.get("objects", []):
            obj = create_object(self.space, obj_config)
            self.objects[obj["id"]] = obj

        for seg_config in self.level_data.get("static_segments", []):
            fixed_seg_config = dict(seg_config)
            fixed_seg_config["type"] = "segment"
            fixed_seg_config["dynamic"] = False

            seg = create_segment(self.space, fixed_seg_config)
            self.static_segments.append(seg)

        self.goal_evaluator = GoalEvaluator(self.level_data, self.objects)
        self.goal_evaluator.reset()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.step_count = 0
        self.goal_reached = False
        self.last_action = None
        self.last_line_length = 0.0

        self._setup_world()

        self.prev_goal_distance = self._distance_to_first_goal()

        obs = self._get_obs()
        info = {
            "level_title": self.level_data.get("title", "Untitled Level"),
            "is_success": False,
        }

        if self.render_mode == "human":
            self.render()
        self.simulation_started = False

        return obs, info

    def step(self, action):
        if self.control_mode == "agent":
            if not self.simulation_started:
                if self.agent_draw_mode == "stroke":
                    draw_success = self._apply_agent_stroke_action(action)
                else:
                    draw_success = self._apply_draw_action(action)

                if not draw_success:
                    obs = self._get_obs()
                    info = {
                        "is_success": False,
                        "is_failure": True,
                        "failure_reason": "invalid_draw",
                        "step_count": self.step_count,
                        "segments_used": len(self.agent_segments),
                        "level_id": self.level_id,
                        "level_title": self.level_data.get("title", "Untitled Level"),
                    }
                    return obs, -50.0, False, True, info
            # elif len(self.agent_segments) < self.max_drawn_segments:
            #     self._apply_agent_stroke_action(action)

        elif self.control_mode == "human":
            pass

        if not self.simulation_started:
            obs = self._get_obs()
            info = {
                "is_success": self.goal_reached,
                "step_count": self.step_count,
                "segments_used": len(self.agent_segments),
                "goal_distance": self.prev_goal_distance,
                "level_id": self.level_id,
                "level_title": self.level_data.get("title", "Untitled Level"),
            }
            return obs, 0.0, False, False, info

        self.step_count += 1

        substeps = int(self.level_data.get("physics", {}).get("substeps", 4))

        if self.space is None:
            raise RuntimeError("Physics space is not initialized. Call reset() before step().")

        for _ in range(substeps):
            self.space.step(self.DT)

            if self.goal_evaluator.check_success():
                self.goal_reached = True
                break

        current_goal_distance = self._distance_to_first_goal()
        reward = self._compute_reward(self.prev_goal_distance, current_goal_distance)
        self.prev_goal_distance = current_goal_distance

   
        terminated = self.goal_reached
        truncated = (self.step_count >= self.max_steps) and not self.goal_reached

        if truncated:
            reward -= 50.0   # clear failure signal for timeout

        obs = self._get_obs()

        info = {
            "is_success": self.goal_reached,
            "is_failure": truncated,
            "step_count": self.step_count,
            "segments_used": len(self.agent_segments),
            "goal_distance": current_goal_distance,
            "level_id": self.level_id,
            "level_title": self.level_data.get("title", "Untitled Level"),
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info
    
    def _apply_agent_stroke_action(self, action):
        if len(self.agent_segments) >= self.max_drawn_segments:
            return False

        action = np.asarray(action, dtype=np.float32).flatten()

        if action.shape[0] != self.num_stroke_points * 2:
            return False

        points = []

        for i in range(self.num_stroke_points):
            x = self._clamp(action[2 * i], self.draw_region["x_min"], self.draw_region["x_max"])
            y = self._clamp(action[2 * i + 1], self.draw_region["y_min"], self.draw_region["y_max"])
            points.append((x, y))

        total_length = 0.0
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            total_length += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if total_length < 25:
            self.last_line_length = total_length
            self.last_action = action
            return False

        self._apply_stroke_points(points)

        if self.render_mode == "human":
            print("PPO drew stroke:", points)

        return True
    
    def _apply_stroke_points(self, points):
        if self.space is None:
            return

        if len(self.agent_segments) >= self.max_drawn_segments:
            return

        if len(points) < 2:
            return

        draw_rules = self.level_data.get("draw_rules", {})
        friction = float(draw_rules.get("friction", 0.9))
        elasticity = float(draw_rules.get("elasticity", 0.3))
        stroke_mass = float(draw_rules.get("stroke_mass", 2.0))

        cleaned_points = []
        min_dist = 5.0

        # Clean and clamp points
        for p in points:
            x = self._clamp(p[0], self.draw_region["x_min"], self.draw_region["x_max"])
            y = self._clamp(p[1], self.draw_region["y_min"], self.draw_region["y_max"])

            if not cleaned_points:
                cleaned_points.append((x, y))
            else:
                px, py = cleaned_points[-1]
                dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)

                if dist >= min_dist:
                    cleaned_points.append((x, y))

        if len(cleaned_points) < 2:
            return

        stroke_shapes = []
        stroke_bodies = []

        # -------------------------------------------------
        # CASE 1: Dynamic stroke
        # One body + many segment shapes
        # -------------------------------------------------
        if self.stroke_body_type == "dynamic":
            center_x = sum(p[0] for p in cleaned_points) / len(cleaned_points)
            center_y = sum(p[1] for p in cleaned_points) / len(cleaned_points)

            local_points = [
                (x - center_x, y - center_y)
                for x, y in cleaned_points
            ]

            min_x = min(p[0] for p in local_points)
            max_x = max(p[0] for p in local_points)
            min_y = min(p[1] for p in local_points)
            max_y = max(p[1] for p in local_points)

            width = max(10.0, max_x - min_x)
            height = max(10.0, max_y - min_y)

            moment = pymunk.moment_for_box(stroke_mass, (width, height))

            body = pymunk.Body(stroke_mass, moment)
            body.position = (center_x, center_y)

            for i in range(len(local_points) - 1):
                local_a = local_points[i]
                local_b = local_points[i + 1]

                shape = pymunk.Segment(
                    body,
                    local_a,
                    local_b,
                    self.agent_segment_thickness / 2.0,
                )

                shape.friction = friction
                shape.elasticity = elasticity

                stroke_shapes.append(shape)

            self.space.add(body, *stroke_shapes)
            stroke_bodies.append(body)

        # -------------------------------------------------
        # CASE 2: Static stroke
        # Static body + many fixed segment shapes
        # -------------------------------------------------
        else:
            body = self.space.static_body

            for i in range(len(cleaned_points) - 1):
                x1, y1 = cleaned_points[i]
                x2, y2 = cleaned_points[i + 1]

                shape = pymunk.Segment(
                    body,
                    (x1, y1),
                    (x2, y2),
                    self.agent_segment_thickness / 2.0,
                )

                shape.friction = friction
                shape.elasticity = elasticity

                self.space.add(shape)
                stroke_shapes.append(shape)

            stroke_bodies.append(body)

        stroke_data_points = [
            {"x": x, "y": y}
            for x, y in cleaned_points
        ]

        stroke_data = {
            "type": "stroke",
            "body_type": self.stroke_body_type,
            "points": stroke_data_points,
            "bodies": stroke_bodies,
            "shapes": stroke_shapes,
            "thickness": self.agent_segment_thickness,
        }

        self.agent_segments.append(stroke_data)

        # Calculate total stroke length
        total_length = 0.0
        for i in range(len(cleaned_points) - 1):
            x1, y1 = cleaned_points[i]
            x2, y2 = cleaned_points[i + 1]
            total_length += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        self.last_line_length = total_length

        self.last_action = np.array(
            [
                cleaned_points[0][0],
                cleaned_points[0][1],
                cleaned_points[-1][0],
                cleaned_points[-1][1],
            ],
            dtype=np.float32,
        )

        self.simulation_started = True
       

    def _apply_draw_action(self, action):
        if len(self.agent_segments) >= self.max_drawn_segments:
            return

        action = np.asarray(action, dtype=np.float32).flatten()
        if action.shape[0] != 4:
            return

        x1, y1, x2, y2 = action.tolist()

        x1 = self._clamp(x1, self.draw_region["x_min"], self.draw_region["x_max"])
        y1 = self._clamp(y1, self.draw_region["y_min"], self.draw_region["y_max"])
        x2 = self._clamp(x2, self.draw_region["x_min"], self.draw_region["x_max"])
        y2 = self._clamp(y2, self.draw_region["y_min"], self.draw_region["y_max"])

        if abs(x1 - x2) < 1e-3 and abs(y1 - y2) < 1e-3:
            x2 = min(self.draw_region["x_max"], x1 + 40.0)

        draw_rules = self.level_data.get("draw_rules", {})
        friction = float(draw_rules.get("friction", 0.9))
        elasticity = float(draw_rules.get("elasticity", 0.3))
        stroke_mass = float(draw_rules.get("stroke_mass", 1.0))

        if self.stroke_body_type == "dynamic":
            mid_x = (x1 + x2) / 2.0
            mid_y = (y1 + y2) / 2.0

            local_a = (x1 - mid_x, y1 - mid_y)
            local_b = (x2 - mid_x, y2 - mid_y)

            moment = pymunk.moment_for_segment(
                stroke_mass,
                local_a,
                local_b,
                self.agent_segment_thickness / 2.0,
            )

            body = pymunk.Body(stroke_mass, moment)
            body.position = (mid_x, mid_y)

            shape = pymunk.Segment(
                body,
                local_a,
                local_b,
                self.agent_segment_thickness / 2.0,
            )

            shape.friction = friction
            shape.elasticity = elasticity

            self.space.add(body, shape)

        else:
            body = self.space.static_body

            shape = pymunk.Segment(
                body,
                (x1, y1),
                (x2, y2),
                self.agent_segment_thickness / 2.0,
            )

            shape.friction = friction
            shape.elasticity = elasticity

            self.space.add(shape)
       

        segment_data = {
            "type": "line",
            "body_type": self.stroke_body_type,
            "body": body,
            "a": {"x": x1, "y": y1},
            "b": {"x": x2, "y": y2},
            "thickness": self.agent_segment_thickness,
            "shape": shape,
        }

        self.agent_segments.append(segment_data)
        self.last_action = action
        self.last_line_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # Start simulation after first valid stroke
        if not self.simulation_started:
            self.simulation_started = True
        print(f"PPO drew line: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")

    def _compute_reward(self, prev_dist, current_dist):
        if self.reward_mode == "sparse":
            return 1.0 if self.goal_reached else -1.0

        reward = -0.02

        progress = prev_dist - current_dist
        reward += 0.05 * progress

        reward -= 0.03 * len(self.agent_segments)

        if self.last_action is not None and self.last_line_length < 25:
            reward -= 1.0

        if current_dist < 200:
            reward += 1.0

        if current_dist < 100:
            reward += 2.0

        if current_dist < 50:
            reward += 5.0

        if self.goal_reached:
            reward += 100.0

        return float(reward)
    def _get_obs(self):
        dynamic_objects = [
            obj for obj in self.objects.values()
            if obj["config"].get("dynamic", True)
        ]

        if not dynamic_objects:
            return np.zeros(10, dtype=np.float32)

        obj = dynamic_objects[0]
        body = obj["body"]

        x = body.position.x / self.canvas_width
        y = body.position.y / self.canvas_height
        vx = np.clip(body.velocity.x / self.MAX_SPEED, -1.0, 1.0)
        vy = np.clip(body.velocity.y / self.MAX_SPEED, -1.0, 1.0)

        gx, gy = self._first_goal_center()

        gx_norm = gx / self.canvas_width
        gy_norm = gy / self.canvas_height

        rel_x = (gx - body.position.x) / self.canvas_width
        rel_y = (gy - body.position.y) / self.canvas_height

        step_norm = min(1.0, self.step_count / max(1, self.max_steps))
        seg_norm = min(1.0, len(self.agent_segments) / max(1, self.max_drawn_segments))

        return np.array(
            [x, y, vx, vy, gx_norm, gy_norm, rel_x, rel_y, step_norm, seg_norm],
            dtype=np.float32,
        )

    def _first_goal_center(self):
        goals = self.level_data.get("goals", [])
        if not goals:
            return 0.0, 0.0

        region = goals[0].get("region", {})
        return float(region.get("x", 0.0)), float(region.get("y", 0.0))

    def _distance_to_first_goal(self):
        goals = self.level_data.get("goals", [])
        if not goals:
            return 0.0

        object_id = goals[0].get("object_id")
        if object_id not in self.objects:
            return 0.0

        obj = self.objects[object_id]
        body = obj["body"]

        gx, gy = self._first_goal_center()

        return math.sqrt((body.position.x - gx) ** 2 + (body.position.y - gy) ** 2)

    def render(self):
        if self.render_mode is None:
            return None

        if self.screen is None:
            pygame.init()
            pygame.font.init()

            if self.render_mode == "human":
                pygame.display.set_caption("BrainItOn General Env")
                self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            else:
                self.screen = pygame.Surface((self.window_width, self.window_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.font is None:
            self.font = pygame.font.SysFont("Arial", 18)

        self.screen.fill((245, 245, 248))


        # ==========================================
        # DRAW ACTION SPACE OVERLAY
        # ==========================================

        draw_region = self.draw_region

        x_min = int(draw_region["x_min"])
        x_max = int(draw_region["x_max"])
        y_min = int(draw_region["y_min"]) + self.ui_height
        y_max = int(draw_region["y_max"]) + self.ui_height

        # Transparent overlay
        overlay = pygame.Surface(
            (self.window_width, self.window_height),
            pygame.SRCALPHA
        )

        # Fill full gameplay area with light red
        pygame.draw.rect(
            overlay,
            (255, 180, 180, 70),
            pygame.Rect(
                0,
                self.ui_height,
                self.window_width,
                self.canvas_height
            )
        )

        # Draw action space as white
        pygame.draw.rect(
            overlay,
            (255, 255, 255, 140),
            pygame.Rect(
                x_min,
                y_min,
                x_max - x_min,
                y_max - y_min
            )
        )

        # Optional blue border
        pygame.draw.rect(
            overlay,
            (50, 120, 255, 255),
            pygame.Rect(
                x_min,
                y_min,
                x_max - x_min,
                y_max - y_min
            ),
            3
        )

        self.screen.blit(overlay, (0, 0))

# ==========================================

        self._draw_goals()
        self._draw_static_segments()
        self._draw_agent_segments()
        self._draw_objects()
        self._draw_hud()

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None

        frame = pygame.surfarray.array3d(self.screen)
        return np.transpose(frame, (1, 0, 2))

    def _draw_goals(self):
        for goal in self.level_data.get("goals", []):
            region = goal.get("region", {})

            if region.get("shape") == "circle":
                pygame.draw.circle(
                    self.screen,
                    (100, 220, 120),
                    (int(region["x"]), int(region["y"] + self.ui_height)),
                    int(region["r"]),
                )

            elif region.get("shape") == "rectangle":
                pygame.draw.rect(
                    self.screen,
                    (100, 220, 120),
                    pygame.Rect(
                        int(region["x"]),
                        int(region["y"] + self.ui_height),
                        int(region["width"]),
                        int(region["height"]),
                    ),
                    width=2,
                )

    def _draw_static_segments(self):
        for seg in self.level_data.get("static_segments", []):
            pygame.draw.line(
                self.screen,
                (60, 60, 60),
                (int(seg["a"]["x"]), int(seg["a"]["y"])+self.ui_height),
                (int(seg["b"]["x"]), int(seg["b"]["y"])+self.ui_height),
                int(seg.get("thickness", 8.0)),
            )

    def _draw_agent_segments(self):
        for seg in self.agent_segments:
            thickness = int(seg.get("thickness", self.agent_segment_thickness))

            if seg.get("type") == "stroke":
                if seg.get("body_type") == "dynamic":
                    for shape in seg.get("shapes", []):
                        body = shape.body

                        a = body.local_to_world(shape.a)
                        b = body.local_to_world(shape.b)

                        pygame.draw.line(
                            self.screen,
                            (40, 120, 220),
                            (int(a.x), int(a.y + self.ui_height)),
                            (int(b.x), int(b.y + self.ui_height)),
                            thickness,
                        )
                else:
                    points = seg.get("points", [])

                    if len(points) >= 2:
                        pygame_points = [
                            (int(p["x"]), int(p["y"] + self.ui_height))
                            for p in points
                        ]

                        pygame.draw.lines(
                            self.screen,
                            (40, 120, 220),
                            False,
                            pygame_points,
                            thickness,
                        )

            else:
                if seg.get("body_type") == "dynamic":
                    shape = seg["shape"]
                    body = seg["body"]

                    a = body.local_to_world(shape.a)
                    b = body.local_to_world(shape.b)

                    pygame.draw.line(
                        self.screen,
                        (40, 120, 220),
                        (int(a.x), int(a.y + self.ui_height)),
                        (int(b.x), int(b.y + self.ui_height)),
                        thickness,
                    )
                else:
                    pygame.draw.line(
                        self.screen,
                        (40, 120, 220),
                        (int(seg["a"]["x"]), int(seg["a"]["y"] + self.ui_height)),
                        (int(seg["b"]["x"]), int(seg["b"]["y"] + self.ui_height)),
                        thickness,
                    )
    def _draw_objects(self):
        for obj in self.objects.values():
            obj_type = obj["type"]
            body = obj["body"]
            config = obj["config"]
            color = tuple(config.get("color", [220, 70, 70]))

            if obj_type == "circle":
                radius = int(config["radius"])
                x = int(body.position.x)
                y = int(body.position.y+self.ui_height)

                pygame.draw.circle(self.screen, color, (x, y), radius)

                # rotation marker
                angle = body.angle
                marker_x = x + int(math.cos(angle) * radius)
                marker_y = y + int(math.sin(angle) * radius)

                pygame.draw.line(
                    self.screen,
                    (255, 255, 255),
                    (x, y),
                    (marker_x, marker_y),
                    4,
                )

            elif obj_type == "box":
                vertices = obj["shape"].get_vertices()
                points = []

                for v in vertices:
                    world_v = body.local_to_world(v)
                    points.append((int(world_v.x), int(world_v.y)+self.ui_height))

                pygame.draw.polygon(self.screen, color, points)
            elif obj_type == "polygon":
                vertices = obj["shape"].get_vertices()
                points = []

                for v in vertices:
                    world_v = body.local_to_world(v)
                    points.append((int(world_v.x), int(world_v.y)))

                pygame.draw.polygon(self.screen, color, points)

    def _draw_hud(self):
        title = f"Level {self.level_id}: {self.level_data.get('title', 'Untitled')}"

        speed = 0.0
        angular_velocity = 0.0

        dynamic_objects = [
            obj for obj in self.objects.values()
            if obj["config"].get("dynamic", True)
        ]

        if dynamic_objects:
            body = dynamic_objects[0]["body"]
            speed = body.velocity.length
            angular_velocity = body.angular_velocity

        physics = self.level_data.get("physics", {})
        gravity = float(physics.get("gravity", 500.0))
        damping = float(physics.get("damping", 0.98))
        substeps = int(physics.get("substeps", 4))

        # Background top bar
        pygame.draw.rect(
            self.screen,
            (245, 245, 248),
            pygame.Rect(0, 0, self.window_width, self.ui_height),
        )

        pygame.draw.line(
            self.screen,
            (180, 180, 180),
            (0, self.ui_height - 1),
            (self.window_width, self.ui_height - 1),
            2,
        )

        title_font = pygame.font.SysFont("Arial", 22, bold=True)
        small_font = pygame.font.SysFont("Arial", 16)

        # Truncate title if too long
        max_title_chars = 55
        if len(title) > max_title_chars:
            title = title[:max_title_chars] + "..."

        title_surface = title_font.render(title, True, (25, 25, 25))
        self.screen.blit(title_surface, (16, 12))

        left_lines = [
            f"Mode: {self.control_mode} | Reward: {self.reward_mode}",
            f"Step: {self.step_count}/{self.max_steps}",
            f"Segments: {len(self.agent_segments)}/{self.max_drawn_segments}",
        ]

        right_lines = [
            f"Success: {self.goal_reached}",
            # f"Speed: {speed:.2f}",
            # f"Angular velocity: {angular_velocity:.2f}",
            # f"Gravity: {gravity:.1f} | Damping: {damping:.3f} | Substeps: {substeps}",
        ]

        y = 45
        for line in left_lines:
            surface = small_font.render(line, True, (35, 35, 35))
            self.screen.blit(surface, (18, y))
            y += 22

        y = 45
        for line in right_lines:
            color = (20, 20, 20)

            if "Success: True" in line:
                color = (0, 150, 0)
            elif "Success: False" in line:
                color = (180, 0, 0)
            surface = small_font.render(line, True, color)
            self.screen.blit(surface, (360, y))
            y += 22

        
        hint = "R: reset | C: clear | N/P: level | Space: pause"

        hint_surface = small_font.render(hint, True, (80, 80, 80))
        self.screen.blit(hint_surface, (18, self.ui_height - 24))

    def clear_drawn_segments(self):
        if self.space is None:
            return

        for seg in self.agent_segments:
            if seg.get("type") == "stroke":
                for shape in seg.get("shapes", []):
                    if shape is not None and shape in self.space.shapes:
                        body = shape.body
                        self.space.remove(shape)

                        if body is not self.space.static_body and body in self.space.bodies:
                            self.space.remove(body)
            else:
                shape = seg.get("shape")
                body = seg.get("body")

                if shape is not None and shape in self.space.shapes:
                    self.space.remove(shape)

                if body is not None and body is not self.space.static_body and body in self.space.bodies:
                    self.space.remove(body)

        self.agent_segments.clear()
        self.current_stroke_points = []
        self.preview_stroke_points = []
        self.simulation_started = False
        
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None

        self.space = None
        self.objects = {}
        self.static_segments = []
        self.agent_segments = []

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, value))
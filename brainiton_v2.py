import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
import pymunk
from gymnasium import spaces
from pathlib import Path
import json


class BrainItOnEnvV2(gym.Env):
    """
    Version 2 of the Brain It On style environment.

    Main upgrades over V1:
    - Multiple levels 
    - Multi-stroke drawing (agent lines are accumulated) ---- Will change to strokes array afterwards
    - Dense and sparse reward modes
    - Human mode with mouse drawing
    - LLM/VLM mode hook 
    - Better info/debug tracking
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # ------------------------------------------------------------
    # Global canvas / physics constants
    # ------------------------------------------------------------
    CANVAS_WIDTH = 900
    CANVAS_HEIGHT = 620

    # IMPORTANT:
    # This is in simulation/canvas units, not real-world SI units.
    GRAVITY = 900.0

    FPS = 60
    DT = 1.0 / FPS  # DT means timestep for physics simulation; it can be decoupled from rendering FPS if needed

    BALL_RADIUS = 18.0
    BALL_RESTITUTION = 0.4
    BALL_FRICTION = 0.95

    DEFAULT_SEGMENT_THICKNESS = 8.0
    AGENT_SEGMENT_THICKNESS = 4.0

    MAX_SPEED = 1200.0

    def __init__(
        self,
        level_id: int = 1,
        render_mode: Optional[str] = None,
        reward_mode: str = "dense",          # "dense" or "sparse"
        control_mode: str = "agent",         # "agent", "human", "llm"
        max_steps: int = 80,
        line_simulation_steps: int = 10,
        max_drawn_segments: int = 5,
        gravity: float = 900.0,
        ball_friction: float = 0.95,
        allow_accumulated_drawing: bool = True,
        surface_friction: float = 0.9,
        levels_path: str = "levels/level.json",   # change the path of the level if you want to
    ) -> None:
        super().__init__()

        self.level_id = level_id
        self.render_mode = render_mode
        self.reward_mode = reward_mode
        self.control_mode = control_mode

        self.max_steps = max_steps
        self.line_simulation_steps = line_simulation_steps
        self.max_drawn_segments = max_drawn_segments
        self.allow_accumulated_drawing = allow_accumulated_drawing

        self.gravity = gravity
        self.ball_friction = ball_friction
        self.surface_friction = surface_friction

        # Pymunk objects
        self.space: Optional[pymunk.Space] = None
        self.ball_body: Optional[pymunk.Body] = None
        self.ball_shape: Optional[pymunk.Circle] = None
        self.static_shapes: List[pymunk.Shape] = []
        self.agent_shapes: List[pymunk.Shape] = []

        # Rendering state
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None

        # Human interaction state
        self.current_stroke_start: Optional[Tuple[int, int]] = None
        self.human_done_drawing = False
        self.human_auto_simulate = True

        # Environment state
        self.level_data: Dict[str, Any] = {}
        self.ball: Dict[str, float] = {}
        self.user_strokes: List[List[Dict[str, float]]] = []
        self.stroke_segments: List[Dict[str, Any]] = []

        self.goal_reached = False
        self.step_count = 0
        self.prev_goal_dist = 0.0

        self.last_line_length = 0.0
        self.last_line_start = (0.0, 0.0)
        self.last_applied_action: Optional[np.ndarray] = None

        self.debug_text = ""

        # Drawing region
        self.draw_region = {
            "x_min": 20.0,
            "x_max": 850.0,
            "y_min": 20.0,
            "y_max": 600.0,
        }

        # Load current level
        self.levels_path = Path(levels_path)
        self.levels = self._load_levels_from_file(self.levels_path)
        self._load_level(self.level_id)

        # Action space:
        # a single line segment = [x1, y1, x2, y2]
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

        # Observation:
        # [ball_x, ball_y, ball_vx, ball_vy,
        #  goal_x, goal_y, rel_goal_x, rel_goal_y,
        #  norm_step_count, norm_segments_used]
        self.observation_space = spaces.Box(
            low=np.array([-1.0] * 10, dtype=np.float32),
            high=np.array([1.0] * 10, dtype=np.float32),
            dtype=np.float32,
        )

    # -------------------------------------------------
    # loading levels
    #--------------------------------------------------

    def _load_levels_from_file(self, levels_path: Path) -> Dict[str, Any]:
        if not levels_path.exists():
            raise FileNotFoundError(f"Levels file not found: {levels_path}")

        with open(levels_path, "r", encoding="utf-8") as f:
            levels = json.load(f)

        if not isinstance(levels, dict):
            raise ValueError("Levels JSON must be a dictionary with level IDs as keys.")

        return levels


    def _load_level(self, level_id: int) -> None:
        level_key = str(level_id)

        if level_key not in self.levels:
            raise ValueError(
                f"Level {level_id} not found in {self.levels_path}. "
                f"Available levels: {list(self.levels.keys())}"
            )

        self.level_data = self.levels[level_key]




    # ------------------------------------------------------------
    # Setup physics world
    # ------------------------------------------------------------
    def _setup_pymunk_world(self) -> None:
        self.space = pymunk.Space()
        self.space.gravity = (0.0, self.gravity)

        self.static_shapes = []
        self.agent_shapes = []

        radius = float(self.level_data["ball"]["r"])
        mass = 1.0
        moment = pymunk.moment_for_circle(mass, 0, radius)

        self.ball_body = pymunk.Body(mass, moment)
        self.ball_body.position = (
            float(self.level_data["ball"]["x"]),
            float(self.level_data["ball"]["y"]),
        )
        self.ball_body.velocity = (0.0, 0.0)

        self.ball_shape = pymunk.Circle(self.ball_body, radius)
        self.ball_shape.elasticity = self.BALL_RESTITUTION
        self.ball_shape.friction = self.ball_friction

        self.space.add(self.ball_body, self.ball_shape)

        static_body = self.space.static_body
        for seg in self.level_data["staticSegments"]:
            shape = pymunk.Segment(
                static_body,
                (seg["a"]["x"], seg["a"]["y"]),
                (seg["b"]["x"], seg["b"]["y"]),
                float(seg.get("thickness", self.DEFAULT_SEGMENT_THICKNESS)) / 2.0,
            )
            shape.elasticity = self.BALL_RESTITUTION
            shape.friction =  shape.friction = float(seg.get("friction", self.surface_friction))
            self.static_shapes.append(shape)

        if self.static_shapes:
            self.space.add(*self.static_shapes)

    # ------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if options:
            if "level_id" in options:
                self.level_id = int(options["level_id"])
                self._load_level(self.level_id)
            if "reward_mode" in options:
                self.reward_mode = str(options["reward_mode"])
            if "control_mode" in options:
                self.control_mode = str(options["control_mode"])

        self.goal_reached = False
        self.step_count = 0
        self.prev_goal_dist = 0.0

        self.user_strokes = []
        self.stroke_segments = []
        self.last_applied_action = None
        self.last_line_length = 0.0
        self.last_line_start = (0.0, 0.0)

        self.current_stroke_start = None
        self.human_done_drawing = False

        self.ball = {
            "x": float(self.level_data["ball"]["x"]),
            "y": float(self.level_data["ball"]["y"]),
            "vx": 0.0,
            "vy": 0.0,
            "r": float(self.level_data["ball"]["r"]),
        }

        self._setup_pymunk_world()
        self.prev_goal_dist = self._goal_distance()

        obs = self._get_obs()
        info = {
            "level_title": self.level_data["levelTitle"],
            "is_success": False,
            "reward_mode": self.reward_mode,
            "control_mode": self.control_mode,
        }

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1

        if self.control_mode == "agent":
            self._apply_action_with_policy(action)

        elif self.control_mode == "llm":
            llm_action = self.sample_llm_action()
            self._apply_action_with_policy(llm_action)

        elif self.control_mode == "human":
            # In human mode, actions are added through mouse interaction.
            # We only simulate here.
            pass

        prev_goal_dist = self.prev_goal_dist

        for _ in range(self.line_simulation_steps):
            self._simulate_one_substep()
            if self.goal_reached:
                break

        current_goal_dist = self._goal_distance()
        reward = self._compute_reward(prev_goal_dist, current_goal_dist)
        self.prev_goal_dist = current_goal_dist

        terminated = self.goal_reached
        truncated = self.step_count >= self.max_steps

        # Optional extra truncation if too many segments are used
        if len(self.agent_shapes) > self.max_drawn_segments:
            truncated = True

        obs = self._get_obs()
        info = {
            "is_success": self.goal_reached,
            "level_title": self.level_data["levelTitle"],
            "step_count": self.step_count,
            "segments_used": len(self.agent_shapes),
            "applied_action": None if self.last_applied_action is None else self.last_applied_action.tolist(),
            "goal_distance": current_goal_dist,
            "reward_mode": self.reward_mode,
            "control_mode": self.control_mode,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------
    # Action handling
    # ------------------------------------------------------------
    def _apply_action_with_policy(self, action: np.ndarray) -> None:
        """
        Applies one line segment action.
        In V2, old lines are kept unless allow_accumulated_drawing=False.
        """
        if self.space is None:
            return

        if not self.allow_accumulated_drawing:
            self._clear_agent_shapes()

        if len(self.agent_shapes) >= self.max_drawn_segments:
            return

        self._apply_action(action)

    def _clear_agent_shapes(self) -> None:
        if self.space is None:
            return

        for shape in self.agent_shapes:
            if shape in self.space.shapes:
                self.space.remove(shape)

        self.agent_shapes.clear()
        self.stroke_segments.clear()
        self.user_strokes.clear()

    def _apply_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32).flatten()

        if action.shape[0] != 4:
            raise ValueError("Action must have shape (4,) => [x1, y1, x2, y2]")

        x1, y1, x2, y2 = action.tolist()

        x1 = self._clamp(x1, self.draw_region["x_min"], self.draw_region["x_max"])
        y1 = self._clamp(y1, self.draw_region["y_min"], self.draw_region["y_max"])
        x2 = self._clamp(x2, self.draw_region["x_min"], self.draw_region["x_max"])
        y2 = self._clamp(y2, self.draw_region["y_min"], self.draw_region["y_max"])

        # Prevent zero-length line
        if abs(x1 - x2) < 1e-3 and abs(y1 - y2) < 1e-3:
            x2 = min(self.draw_region["x_max"], x1 + 40.0)

        points = [{"x": x1, "y": y1}, {"x": x2, "y": y2}]
        self.user_strokes.append(points)
        self.stroke_segments.append(
            {
                "a": points[0],
                "b": points[1],
                "thickness": self.AGENT_SEGMENT_THICKNESS,
            }
        )

        self.last_line_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        self.last_line_start = (x1, y1)
        self.last_applied_action = np.array([x1, y1, x2, y2], dtype=np.float32)

        if self.space is not None:
            shape = pymunk.Segment(
                self.space.static_body,
                (x1, y1),
                (x2, y2),
                self.AGENT_SEGMENT_THICKNESS / 2.0,
            )
            shape.elasticity = self.BALL_RESTITUTION
            shape.friction = 0.9
            self.agent_shapes.append(shape)
            self.space.add(shape)

    # ------------------------------------------------------------
    # Human mode
    # ------------------------------------------------------------
    def run_human_mode(self) -> None:
        """
        Run an interactive human mode using mouse drawing.
        Left mouse:
            click-drag-release to draw one segment
        Keys:
            R = reset level
            C = clear drawn lines
            N = next level
            P = previous level
            SPACE = toggle pause/simulate
            ESC = quit
        """
        self.control_mode = "human"
        self.render_mode = "human"

        obs, info = self.reset()
        print("Human mode started.")
        print("Level:", info["level_title"])
        print("Controls: drag mouse to draw | R reset | C clear | N next | P prev | SPACE pause/resume | ESC quit")

        running = True
        simulate = True

        while running:
            if self.screen is None:
                self.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_r:
                        self.reset()

                    elif event.key == pygame.K_c:
                        self._clear_agent_shapes()

                    elif event.key == pygame.K_n:
                        available_levels = sorted([int(k) for k in self.levels.keys()])
                        current_index = available_levels.index(self.level_id)
                        next_index = (current_index + 1) % len(available_levels)
                        self.level_id = available_levels[next_index]
                        self._load_level(self.level_id)
                        self.reset()

                    elif event.key == pygame.K_p:
                        available_levels = sorted([int(k) for k in self.levels.keys()])
                        current_index = available_levels.index(self.level_id)
                        prev_index = (current_index - 1) % len(available_levels)
                        self.level_id = available_levels[prev_index]
                        self._load_level(self.level_id)
                        self.reset()

                    elif event.key == pygame.K_SPACE:
                        simulate = not simulate

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.current_stroke_start = event.pos

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and self.current_stroke_start is not None:
                        if len(self.agent_shapes) < self.max_drawn_segments:
                            x1, y1 = self.current_stroke_start
                            x2, y2 = event.pos
                            action = np.array([x1, y1, x2, y2], dtype=np.float32)
                            self._apply_action(action)
                        self.current_stroke_start = None

            if simulate and not self.goal_reached:
                self.step(np.zeros(4, dtype=np.float32))  # dummy action; human lines already applied

            self.render()

        self.close()

    # ------------------------------------------------------------
    # LLM / VLM placeholder
    # ------------------------------------------------------------
    def get_state_description(self) -> Dict[str, Any]:
        """
        Returns a structured state description.
        This can later be given to an LLM or paired with a rendered image for a VLM.
        """
        return {
            "ball": {
                "x": self.ball["x"],
                "y": self.ball["y"],
                "vx": self.ball["vx"],
                "vy": self.ball["vy"],
                "r": self.ball["r"],
            },
            "goal": self.level_data["goal"],
            "draw_region": self.draw_region,
            "static_segments": self.level_data["staticSegments"],
            "drawn_segments": self.stroke_segments,
            "segments_remaining": max(0, self.max_drawn_segments - len(self.agent_shapes)),
            "level_title": self.level_data["levelTitle"],
        }

    def sample_llm_action(self) -> np.ndarray:
        """
        Placeholder for an LLM/VLM generated action.
        For now it uses a simple heuristic policy.

        Later you can replace this with:
        - an LLM that outputs [x1, y1, x2, y2]
        - a VLM that looks at a rendered image
        - a stroke model like sketch-rnn style logic
        """
        bx = self.ball["x"]
        by = self.ball["y"]
        gx = self.level_data["goal"]["x"]
        gy = self.level_data["goal"]["y"]

        # Simple heuristic:
        # draw a sloped support slightly below the ball and in the direction of the goal
        direction = 1.0 if gx > bx else -1.0

        x1 = bx - 20.0
        y1 = by + 55.0
        x2 = bx + direction * 180.0
        y2 = by + 95.0

        action = np.array([x1, y1, x2, y2], dtype=np.float32)
        return action

    # ------------------------------------------------------------
    # Simulation and reward
    # ------------------------------------------------------------
    def _simulate_one_substep(self) -> None:
        if self.goal_reached or self.space is None or self.ball_body is None:
            return

        self.space.step(self.DT)

        self.ball["x"] = float(self.ball_body.position.x)
        self.ball["y"] = float(self.ball_body.position.y)
        self.ball["vx"] = float(self.ball_body.velocity.x)
        self.ball["vy"] = float(self.ball_body.velocity.y)

        gx = self.level_data["goal"]["x"]
        gy = self.level_data["goal"]["y"]
        gr = self.level_data["goal"]["r"]

        if self._distance(self.ball["x"], self.ball["y"], gx, gy) <= self.ball["r"] + gr:
            self.goal_reached = True

    def _compute_reward(self, prev_goal_dist: float, current_goal_dist: float) -> float:
        """
        Dense reward:
            small step penalty
            progress reward
            penalty for very short lines
            small usage penalty
            success bonus

        Sparse reward:
            +1 if goal reached
            -1 otherwise
        """
        if self.reward_mode == "sparse":
            return 1.0 if self.goal_reached else -1.0

        reward = -0.05

        # Progress-based reward
        progress = prev_goal_dist - current_goal_dist
        reward += 0.02 * progress

        # Penalize tiny lines
        if self.last_applied_action is not None and self.last_line_length < 25:
            reward -= 1.0

        # Penalize too many drawn segments to encourage efficiency
        reward -= 0.05 * len(self.agent_shapes)

        # Encourage drawing somewhere useful, not too far from the ball
        if self.last_applied_action is not None:
            bx = self.ball["x"]
            by = self.ball["y"]
            lx, ly = self.last_line_start
            line_start_dist = self._distance(bx, by, lx, ly)
            reward -= 0.0015 * line_start_dist

        if self.goal_reached:
            reward += 100.0

        return float(reward)

    # ------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        bx = self.ball["x"] / self.CANVAS_WIDTH
        by = self.ball["y"] / self.CANVAS_HEIGHT

        bvx = np.clip(self.ball["vx"] / self.MAX_SPEED, -1.0, 1.0)
        bvy = np.clip(self.ball["vy"] / self.MAX_SPEED, -1.0, 1.0)

        gx = self.level_data["goal"]["x"] / self.CANVAS_WIDTH
        gy = self.level_data["goal"]["y"] / self.CANVAS_HEIGHT

        rgx = (self.level_data["goal"]["x"] - self.ball["x"]) / self.CANVAS_WIDTH
        rgy = (self.level_data["goal"]["y"] - self.ball["y"]) / self.CANVAS_HEIGHT

        step_norm = min(1.0, self.step_count / max(1, self.max_steps))
        seg_norm = min(1.0, len(self.agent_shapes) / max(1, self.max_drawn_segments))

        obs = np.array(
            [bx, by, bvx, bvy, gx, gy, rgx, rgy, step_norm, seg_norm],
            dtype=np.float32,
        )
        return obs

    def _goal_distance(self) -> float:
        return self._distance(
            self.ball["x"],
            self.ball["y"],
            self.level_data["goal"]["x"],
            self.level_data["goal"]["y"],
        )

    def _get_all_segments(self) -> List[Dict[str, Any]]:
        return self.level_data["staticSegments"] + self.stroke_segments

    # ------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------
    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None

        if self.screen is None:
            pygame.init()
            pygame.font.init()

            if self.render_mode == "human":
                pygame.display.set_caption("BrainItOnEnv V2")
                self.screen = pygame.display.set_mode((self.CANVAS_WIDTH, self.CANVAS_HEIGHT))
            else:
                self.screen = pygame.Surface((self.CANVAS_WIDTH, self.CANVAS_HEIGHT))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.font is None:
            self.font = pygame.font.SysFont("Arial", 18)

        if self.render_mode != "human":
            # In rgb_array mode we do not consume pygame events here.
            pass

        self.screen.fill((245, 245, 248))

        # Draw draw-region box
        pygame.draw.rect(
            self.screen,
            (220, 220, 230),
            pygame.Rect(
                self.draw_region["x_min"],
                self.draw_region["y_min"],
                self.draw_region["x_max"] - self.draw_region["x_min"],
                self.draw_region["y_max"] - self.draw_region["y_min"],
            ),
            width=2,
        )

        # Static segments
        for segment in self.level_data["staticSegments"]:
            pygame.draw.line(
                self.screen,
                (60, 60, 60),
                (int(segment["a"]["x"]), int(segment["a"]["y"])),
                (int(segment["b"]["x"]), int(segment["b"]["y"])),
                int(segment.get("thickness", self.DEFAULT_SEGMENT_THICKNESS)),
            )

        # Agent / human / llm drawn segments
        for segment in self.stroke_segments:
            pygame.draw.line(
                self.screen,
                (40, 120, 220),
                (int(segment["a"]["x"]), int(segment["a"]["y"])),
                (int(segment["b"]["x"]), int(segment["b"]["y"])),
                int(segment.get("thickness", self.AGENT_SEGMENT_THICKNESS)),
            )

        # Preview line while dragging in human mode
        if self.control_mode == "human" and self.current_stroke_start is not None and self.render_mode == "human":
            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.line(
                self.screen,
                (120, 160, 250),
                self.current_stroke_start,
                mouse_pos,
                2,
            )

        # Goal
        goal_color = (100, 220, 120) if not self.goal_reached else (60, 180, 80)
        pygame.draw.circle(
            self.screen,
            goal_color,
            (int(self.level_data["goal"]["x"]), int(self.level_data["goal"]["y"])),
            int(self.level_data["goal"]["r"]),
        )

        # Ball
        # Ball with visible rotation marker
        ball_x = int(self.ball["x"])
        ball_y = int(self.ball["y"])
        ball_r = int(self.ball["r"])

        pygame.draw.circle(
            self.screen,
            (220, 70, 70),
            (ball_x, ball_y),
            ball_r,
        )

        # Draw outline
        pygame.draw.circle(
            self.screen,
            (80, 30, 30),
            (ball_x, ball_y),
            ball_r,
            width=2,
        )

        # Draw rotating marker line to show rolling
        if self.ball_body is not None:
            angle = self.ball_body.angle

            marker_x = ball_x + int(math.cos(angle) * ball_r)
            marker_y = ball_y + int(math.sin(angle) * ball_r)

            pygame.draw.line(
                self.screen,
                (255, 255, 255),
                (ball_x, ball_y),
                (marker_x, marker_y),
                4,
            )

            # Optional second marker for clearer rotation
            marker_x2 = ball_x + int(math.cos(angle + math.pi / 2) * ball_r * 0.65)
            marker_y2 = ball_y + int(math.sin(angle + math.pi / 2) * ball_r * 0.65)

            pygame.draw.line(
                self.screen,
                (255, 255, 255),
                (ball_x, ball_y),
                (marker_x2, marker_y2),
                3,
            )

        # HUD text
        hud_lines = [
            f"{self.level_data['levelTitle']}",
            f"Control: {self.control_mode} | Reward: {self.reward_mode}",
            f"Step: {self.step_count}/{self.max_steps}",
            f"Segments: {len(self.agent_shapes)}/{self.max_drawn_segments}",
            f"Goal reached: {self.goal_reached}",
            f"Gravity: {self.gravity:.1f} | Ball friction: {self.ball_friction:.2f}",
        ]

        y = 10
        for line in hud_lines:
            text_surface = self.font.render(line, True, (30, 30, 30))
            self.screen.blit(text_surface, (10, y))
            y += 22

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None

        frame = pygame.surfarray.array3d(self.screen)
        return np.transpose(frame, (1, 0, 2))

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------
    def close(self) -> None:
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None

        self.space = None
        self.ball_body = None
        self.ball_shape = None
        self.static_shapes = []
        self.agent_shapes = []

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    @staticmethod
    def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))


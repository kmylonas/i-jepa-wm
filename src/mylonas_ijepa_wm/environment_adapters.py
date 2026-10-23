"""Small runtime adapters for PointMaze and DINO-WM's Wall simulator."""

from dataclasses import dataclass
import importlib
from pathlib import Path
import random
import sys
import types

import numpy as np
import torch
import torch.nn.functional as functional

from src.mylonas_ijepa_wm.environment_configs import EnvironmentConfig


@dataclass
class StepResult:
    state: np.ndarray
    frame: np.ndarray
    success: bool
    terminated: bool = False
    truncated: bool = False


def _frame(image):
    array = np.asarray(image)
    if array.shape != (224, 224, 3):
        raise ValueError(f"Expected 224x224 RGB frame, got {array.shape}")
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


class PointMazeAdapter:
    def __init__(self, config: EnvironmentConfig, max_micro_steps: int):
        from src.mylonas_ijepa_wm import plan_pointmaze as pointmaze

        self.base = pointmaze
        self.config = config
        self.env = pointmaze.create_environment(
            config.pointmaze_environment_id,
            max_micro_steps,
            camera="topdown",
            physics_frame_skip=config.pointmaze_physics_frame_skip,
        )
        self.action_low = np.asarray(self.env.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(self.env.action_space.high, dtype=np.float32)
        self.control_dt_seconds = float(self.env.unwrapped.point_env.dt)
        self.goal = None
        self.observation = None

    def reset(self, seed):
        self.observation, info = self.env.reset(seed=seed)
        self.base.hide_goal_marker(self.env)
        self.goal = np.asarray(self.observation["desired_goal"][:2], dtype=np.float32)
        return StepResult(self.state(), self.render(), bool(info.get("success", False)))

    def state(self):
        return np.asarray(self.observation["observation"], dtype=np.float32).copy()

    def render(self):
        return _frame(self.env.render())

    def render_goal(self):
        return _frame(self.base.render_target_observation(self.env))

    def render_video(self):
        return _frame(self.base.render_visualization_frame(self.env))

    def step(self, action):
        self.observation, _, terminated, truncated, info = self.env.step(
            np.asarray(action, dtype=np.float32)
        )
        return StepResult(
            self.state(), self.render(), bool(info.get("success", False)),
            bool(terminated), bool(truncated),
        )

    def close(self):
        self.env.close()


class WallAdapter:
    def __init__(self, config: EnvironmentConfig, max_micro_steps: int):
        if not config.dino_wm_root:
            raise ValueError("Wall requires dino_wm_root in --env-config")
        root = Path(config.dino_wm_root).expanduser().resolve()
        if not (root / "env/wall/envs/wall.py").is_file():
            raise FileNotFoundError(f"DINO-WM Wall source not found in {root}")
        # Loading the whole DINO-WM `env` package imports D4RL PointMaze as a
        # side effect. Mount only its Wall source under a private package name.
        package_name = "_ijepa_dino_wall"
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(root / "env/wall")]
            sys.modules[package_name] = package
        try:
            WallDatasetConfig = importlib.import_module(
                f"{package_name}.data.wall"
            ).WallDatasetConfig
            DotWall = importlib.import_module(f"{package_name}.envs.wall").DotWall
        except ImportError as error:
            raise RuntimeError(
                "DINO-WM Wall needs its Python dependencies (gym, scipy, omegaconf, pyyaml)"
            ) from error
        self.WallDatasetConfig = WallDatasetConfig
        self.DotWall = DotWall
        self.env = None
        self.config = config
        self.action_low = np.full(2, -config.wall_action_limit, dtype=np.float32)
        self.action_high = np.full(2, config.wall_action_limit, dtype=np.float32)
        self.control_dt_seconds = None
        self.goal = None
        self.layout = None

    def _make_environment(self, wall_location, door_location, seed):
        wall_config = self.WallDatasetConfig(
            device="cpu", img_size=65, dot_std=1.7, border_wall_loc=5,
            fix_wall=True, fix_wall_location=wall_location,
            fix_door_location=door_location, wall_width=6, door_space=4,
            wall_padding=20, door_padding=10,
        )
        self.env = self.DotWall(
            rng=np.random.default_rng(seed), wall_config=wall_config,
            fix_wall=True, fix_wall_location=wall_location,
            fix_door_location=door_location, device="cpu",
        )

    def _render_at(self, position):
        dot = self.env._render_dot(torch.as_tensor(position, dtype=torch.float32))
        rgb = self.env.channels_to_img(self.env.wall_img, dot).float().unsqueeze(0)
        image = functional.interpolate(
            rgb, size=(224, 224), mode="bilinear", align_corners=False
        )[0].permute(1, 2, 0).cpu().numpy()
        return _frame(image)

    def reset(self, seed):
        random.seed(seed)
        layouts = self.config.wall_layouts or [[
            self.config.wall_location, self.config.door_location
        ]]
        self.layout = tuple(layouts[seed % len(layouts)])
        self._make_environment(*self.layout, seed)
        self.env.seed(seed)
        start, goal = self.env.generate_random_state(seed)
        self.env.set_init_state(np.asarray(start, dtype=np.float32))
        _, state = self.env.reset()
        self.goal = np.asarray(goal, dtype=np.float32)
        return StepResult(state.detach().cpu().numpy().astype(np.float32), self.render(), False)

    def state(self):
        return self.env.dot_position.detach().cpu().numpy().astype(np.float32).copy()

    def render(self):
        return self._render_at(self.env.dot_position)

    def render_goal(self):
        return self._render_at(self.goal)

    def render_video(self):
        frame = self.render().copy()
        # Show the goal in videos only. Encoder inputs and goal-reference
        # images still contain the original single red dot.
        scale = 223.0 / (self.env.img_size - 1)
        goal_x, goal_y = np.rint(self.goal * scale).astype(int)
        yy, xx = np.ogrid[:224, :224]
        radius_squared = (xx - goal_x) ** 2 + (yy - goal_y) ** 2
        ring = (radius_squared >= 5 ** 2) & (radius_squared <= 8 ** 2)
        frame[ring] = (0, 210, 0)
        return frame

    def step(self, action):
        self.env.step(torch.as_tensor(action, dtype=torch.float32))
        state = self.state()
        success = np.linalg.norm(state[:2] - self.goal) < self.config.wall_success_threshold
        return StepResult(state, self.render(), bool(success))

    def close(self):
        pass


def create_adapter(config: EnvironmentConfig, max_micro_steps: int):
    if config.env == "pointmaze":
        return PointMazeAdapter(config, max_micro_steps)
    if config.env == "wall":
        return WallAdapter(config, max_micro_steps)
    raise ValueError(f"Unsupported environment: {config.env}")

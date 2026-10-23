"""Environment settings shared by collection, embedding, training, and planning."""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path


@dataclass(frozen=True)
class EnvironmentConfig:
    env: str = "pointmaze"
    actions_per_model_step: int = 5
    pointmaze_environment_id: str = "PointMaze_UMaze-v3"
    pointmaze_physics_frame_skip: int = 1
    wall_location: int = 32
    door_location: int = 30
    wall_layouts: list[list[int]] | None = None
    wall_action_limit: float = 1.0
    wall_success_threshold: float = 4.5
    dino_wm_root: str | None = None

    @property
    def state_fields(self):
        return ("x", "y", "vx", "vy") if self.env == "pointmaze" else ("x", "y")

    @property
    def micro_action_dim(self):
        return 2

    @property
    def macro_action_dim(self):
        return self.actions_per_model_step * self.micro_action_dim

    def portable_metadata(self):
        """Record dynamics/layout, excluding the machine-specific DINO checkout path."""
        values = asdict(self)
        values.pop("dino_wm_root")
        return values


def load_environment_config(env="pointmaze", path=None):
    if env not in {"pointmaze", "wall"}:
        raise ValueError(f"Unsupported environment: {env}")
    settings = {}
    if path is not None:
        settings = json.loads(Path(path).read_text())
        if not isinstance(settings, dict):
            raise ValueError("environment config must be a JSON object")
        unknown = set(settings) - set(EnvironmentConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown environment settings: {sorted(unknown)}")
        if "env" in settings and settings["env"] != env:
            raise ValueError("--env and environment config disagree")
    config = replace(EnvironmentConfig(env=env), **settings)
    if config.actions_per_model_step < 1 or config.pointmaze_physics_frame_skip < 1:
        raise ValueError("action and physics step counts must be positive")
    if not 20 <= config.wall_location <= 44 or not 10 <= config.door_location <= 54:
        raise ValueError("Wall location or door location is outside the supported layout range")
    if config.wall_layouts is not None:
        if not isinstance(config.wall_layouts, list) or not config.wall_layouts:
            raise ValueError("wall_layouts must be a nonempty list of [wall_x, door_y]")
        for layout in config.wall_layouts:
            if (not isinstance(layout, list) or len(layout) != 2
                    or not all(isinstance(value, int) for value in layout)
                    or not 20 <= layout[0] <= 44 or not 10 <= layout[1] <= 54):
                raise ValueError("Invalid wall layout; expected [wall_x, door_y]")
    if config.wall_action_limit <= 0 or config.wall_success_threshold <= 0:
        raise ValueError("Wall action limit and success threshold must be positive")
    return config

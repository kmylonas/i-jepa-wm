"""Collect raw image, action, and state episodes for a selected environment."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.mylonas_ijepa_wm.environment_adapters import create_adapter
from src.mylonas_ijepa_wm.environment_configs import load_environment_config
from src.mylonas_ijepa_wm.wall_layout_manifest import load_manifest, manifest_digest


DINO_WALL_POLICY_PARAMETERS = {
    "angle_noise": 0.2,
    "length_mean": 1.0,
    "length_std": 0.4,
    "length_min": 0.2,
    "length_max": 1.8,
    "component_clip": None,
}


def sample_dino_wall_actions(rng, steps):
    """Sample a persistent random heading and truncated-normal action lengths."""
    if steps < 1:
        raise ValueError("steps must be positive")
    initial_angle = rng.uniform(0.0, 2.0 * np.pi)
    parameters = DINO_WALL_POLICY_PARAMETERS
    turns = rng.vonmises(0.0, 1.0 / parameters["angle_noise"], size=steps - 1)
    angles = initial_angle + np.concatenate(([0.0], np.cumsum(turns)))

    mean, std = parameters["length_mean"], parameters["length_std"]
    minimum, maximum = parameters["length_min"], parameters["length_max"]
    lengths = rng.normal(mean, std, size=steps)
    outside = (lengths < minimum) | (lengths > maximum)
    while outside.any():
        lengths[outside] = rng.normal(mean, std, size=outside.sum())
        outside = (lengths < minimum) | (lengths > maximum)

    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    return (directions * lengths[:, None]).astype(np.float32)


def wall_door_action(position, target, layout, action_low, action_high, rng):
    """Steer through the door to collect transitions on both sides of a wall."""
    position = np.asarray(position[:2], dtype=np.float32)
    target = np.asarray(target[:2], dtype=np.float32)
    wall_x, door_y = layout
    target_side = -1.0 if target[0] < wall_x else 1.0
    current_side = -1.0 if position[0] < wall_x else 1.0
    if current_side == target_side and abs(position[0] - wall_x) > 5:
        waypoint = target
    elif abs(position[1] - door_y) < 2 and abs(position[0] - wall_x) < 6:
        waypoint = np.array([wall_x + target_side * 8, door_y], dtype=np.float32)
    else:
        waypoint = np.array([wall_x + current_side * 6, door_y], dtype=np.float32)
    action = (waypoint - position) / 2.0
    action += rng.normal(0, 0.06, size=2)
    return np.clip(action, action_low, action_high).astype(np.float32)


def collect_episode(
    adapter, config, episode_index, steps_per_episode, seed,
    wall_policy="random", layout=None, manifest_sha256=None,
):
    rng = np.random.default_rng(seed)
    initial = adapter.reset(seed, layout=layout) if layout is not None else adapter.reset(seed)
    frames = [initial.frame]
    states = [initial.state]
    actions = []
    use_door_policy = config.env == "wall" and (
        wall_policy == "door" or (wall_policy == "mixed" and episode_index % 2 == 1)
    )
    use_dino_policy = config.env == "wall" and wall_policy == "dino"
    if wall_policy == "wide":
        raise NotImplementedError("wide Wall policy is added in the coverage-audit task")
    dino_actions = sample_dino_wall_actions(rng, steps_per_episode) if use_dino_policy else None
    collection_policy = "random"
    if use_door_policy:
        collection_policy = "door"
    elif use_dino_policy:
        collection_policy = "dino"
    destination = np.asarray(adapter.goal, dtype=np.float32).copy()
    start_position = np.asarray(initial.state[:2], dtype=np.float32).copy()
    for step_index in range(steps_per_episode):
        if use_door_policy:
            if np.linalg.norm(states[-1][:2] - destination) < config.wall_success_threshold:
                destination = start_position if np.array_equal(destination, adapter.goal) else adapter.goal
            action = wall_door_action(
                states[-1], destination, adapter.layout,
                adapter.action_low, adapter.action_high, rng,
            )
        elif use_dino_policy:
            action = dino_actions[step_index]
        else:
            action = rng.uniform(adapter.action_low, adapter.action_high).astype(np.float32)
        outcome = adapter.step(action)
        if (outcome.terminated or outcome.truncated) and step_index + 1 < steps_per_episode:
            raise RuntimeError(f"Episode {episode_index} ended before {steps_per_episode} actions")
        frames.append(outcome.frame)
        states.append(outcome.state)
        actions.append(action)
    return {
        "frames": torch.from_numpy(np.stack(frames)),
        "actions": torch.from_numpy(np.stack(actions)),
        "states": torch.from_numpy(np.stack(states)),
        "desired_goal": torch.as_tensor(adapter.goal, dtype=torch.float32),
        "episode_index": episode_index,
        "seed": seed,
        "manifest_sha256": manifest_sha256,
        "environment": config.env,
        "environment_config": config.portable_metadata(),
        "wall_layout": getattr(adapter, "layout", None),
        "collection_policy": collection_policy,
        "collection_policy_parameters": (
            DINO_WALL_POLICY_PARAMETERS if use_dino_policy else None
        ),
        "state_fields": config.state_fields,
        "actions_per_model_step": config.actions_per_model_step,
        "physics_frame_skip": (
            config.pointmaze_physics_frame_skip if config.env == "pointmaze" else 1
        ),
        "control_dt_seconds": adapter.control_dt_seconds,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["pointmaze", "wall"], default="pointmaze")
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=2000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--steps-per-episode", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--wall-policy", choices=["random", "mixed", "door", "dino"], default="random",
        help="Wall exploration policy; dino uses persistent random headings and variable action lengths.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--episode-manifest", type=Path,
                        help="Explicit per-episode layout, seed, split, and policy assignments.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_environment_config(args.env, args.env_config)
    episode_manifest = load_manifest(args.episode_manifest) if args.episode_manifest else None
    if episode_manifest and args.env != "wall":
        raise ValueError("--episode-manifest requires --env wall")
    if episode_manifest:
        expected_layouts = {
            tuple(layout) for split in ("train", "val", "test")
            for layout in episode_manifest[f"{split}_layouts"]
        }
        if {tuple(layout) for layout in config.wall_layouts or []} != expected_layouts:
            raise ValueError("Environment config wall_layouts disagree with episode manifest")
        if args.steps_per_episode != episode_manifest["steps_per_episode"]:
            raise ValueError("steps-per-episode differs from episode manifest")
    if args.num_episodes < 1 or args.steps_per_episode < 1:
        raise ValueError("episode counts and lengths must be positive")
    num_episodes = len(episode_manifest["episodes"]) if episode_manifest else args.num_episodes
    end_index = num_episodes if args.end_index is None else args.end_index
    if not 0 <= args.start_index < end_index <= num_episodes:
        raise ValueError("Require 0 <= start-index < end-index <= num-episodes")
    if args.steps_per_episode % config.actions_per_model_step:
        raise ValueError("steps-per-episode must divide into whole model steps")
    if args.env != "wall" and args.wall_policy != "random":
        raise ValueError("--wall-policy only applies to --env wall")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "environment_config": config.portable_metadata(),
        "num_episodes": num_episodes,
        "steps_per_episode": args.steps_per_episode,
        "base_seed": None if episode_manifest else args.base_seed,
        "image_shape": [224, 224, 3],
        "state_fields": config.state_fields,
        "wall_policy": "manifest" if episode_manifest else args.wall_policy,
        "manifest_sha256": manifest_digest(episode_manifest) if episode_manifest else None,
    }
    if args.wall_policy == "dino":
        manifest["wall_policy_parameters"] = DINO_WALL_POLICY_PARAMETERS
    manifest_path = args.output_dir / "metadata.json"
    existing = list(args.output_dir.glob("episode_*.pt"))
    if existing and not args.resume:
        raise FileExistsError("Output contains episodes; use --resume or another directory")
    if args.resume and manifest_path.exists():
        if json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError("Existing dataset uses different environment settings")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    adapter = create_adapter(config, args.steps_per_episode)
    try:
        for episode_index in range(args.start_index, end_index):
            path = args.output_dir / f"episode_{episode_index:04d}.pt"
            assignment = episode_manifest["episodes"][episode_index] if episode_manifest else None
            if args.resume and path.is_file():
                if assignment:
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    if (saved.get("manifest_sha256") != manifest["manifest_sha256"]
                            or saved.get("seed") != assignment["seed"]
                            or tuple(saved.get("wall_layout") or ()) != tuple(assignment["wall_layout"])):
                        raise ValueError(f"Episode {episode_index} differs from episode manifest")
                continue
            episode = collect_episode(
                adapter, config, episode_index, args.steps_per_episode,
                assignment["seed"] if assignment else args.base_seed + episode_index,
                assignment["collection_policy"] if assignment else args.wall_policy,
                layout=assignment["wall_layout"] if assignment else None,
                manifest_sha256=manifest["manifest_sha256"],
            )
            temporary = path.with_suffix(".tmp")
            torch.save(episode, temporary)
            temporary.replace(path)
            if (episode_index + 1) % 10 == 0 or episode_index + 1 == end_index:
                print(f"Saved through episode {episode_index + 1}/{num_episodes} ({args.env})")
    finally:
        adapter.close()


if __name__ == "__main__":
    main()

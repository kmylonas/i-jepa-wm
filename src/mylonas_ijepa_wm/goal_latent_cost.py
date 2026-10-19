"""Measure planner latent MSE to fixed, rendered PointMaze goals."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.mylonas_ijepa_wm.embed_frames import preprocess_frames
from src.mylonas_ijepa_wm.evaluate_position_probe_grid import (
    generate_valid_grid_positions,
    maze_limits_from_walls,
)
from src.mylonas_ijepa_wm.plan_pointmaze import (
    DEFAULT_ENCODER_CHECKPOINT,
    create_environment,
    extract_wall_rectangles,
    hide_goal_marker,
    load_frozen_ijepa_encoder,
    render_target_observation,
    validate_frame,
)
from src.mylonas_ijepa_wm.wm import resolve_device, resolve_precision


DEFAULT_EDGES = (0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 4.0)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "debug" / "goal_latent_cost"


def select_goal_seeds(env, num_goals, base_seed, pool_size=None):
    """Choose diverse valid simulator goals from a small seeded pool."""
    if num_goals < 1:
        raise ValueError("num_goals must be positive")
    pool_size = max(30, 8 * num_goals) if pool_size is None else pool_size
    if pool_size < num_goals:
        raise ValueError("pool_size cannot be smaller than num_goals")
    seeds = list(range(base_seed, base_seed + pool_size))
    goals = []
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        goals.append(np.asarray(observation["desired_goal"], dtype=np.float32))
    goals = np.stack(goals)
    chosen = [0]
    while len(chosen) < num_goals:
        distances = np.linalg.norm(goals[:, None, :] - goals[chosen][None, :, :], axis=-1)
        nearest = distances.min(axis=1)
        nearest[chosen] = -1.0
        chosen.append(int(nearest.argmax()))
    return [seeds[index] for index in chosen]


def sample_positions(goal, valid_positions, edges, per_bin, rng):
    """Include the exact goal and sample collision-free grid positions by distance."""
    goal = np.asarray(goal, dtype=np.float32)
    valid_positions = np.asarray(valid_positions, dtype=np.float32)
    if goal.shape != (2,) or valid_positions.ndim != 2 or valid_positions.shape[1] != 2:
        raise ValueError("goal and valid positions must have shapes [2] and [N, 2]")
    if per_bin < 1 or len(edges) < 2 or edges[0] != 0 or any(
        right <= left for left, right in zip(edges, edges[1:])
    ):
        raise ValueError("invalid distance bins or per-bin sample count")

    distances = np.linalg.norm(valid_positions - goal, axis=1)
    selected = [goal]
    bin_ids = [-1]
    for bin_index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        eligible = np.flatnonzero((distances > lower) & (distances <= upper))
        if len(eligible):
            indices = rng.choice(eligible, size=min(per_bin, len(eligible)), replace=False)
            selected.extend(valid_positions[indices])
            bin_ids.extend([bin_index] * len(indices))
    return np.asarray(selected, dtype=np.float32), bin_ids


def render_positions(env, positions):
    """Render the green ball at requested positions without changing the env state."""
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0:
        raise ValueError("positions must have shape [N, 2] with N > 0")
    point_env = env.unwrapped.point_env
    original_qpos = point_env.data.qpos.copy()
    original_qvel = point_env.data.qvel.copy()
    frames = []
    try:
        for position in positions:
            qpos = original_qpos.copy()
            qpos[:2] = position
            point_env.set_state(qpos, np.zeros_like(original_qvel))
            frames.append(validate_frame(env.render()).copy())
    finally:
        point_env.set_state(original_qpos, original_qvel)
    return np.stack(frames)


def encode_frames(encoder, frames, device, precision, batch_size):
    """Encode a small batch of the exact rendered images used for cost checks."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    latents = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            images = preprocess_frames(torch.from_numpy(frames[start:start + batch_size]))
            images = images.to(device=device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
            ):
                output = encoder(images)
            latents.append(output.float().cpu().numpy())
    return np.concatenate(latents, axis=0)


def save_plots(rows, output_dir, edges, num_goals):
    columns = min(3, num_goals)
    figure, axes = plt.subplots(
        (num_goals + columns - 1) // columns,
        columns,
        figsize=(5 * columns, 4 * ((num_goals + columns - 1) // columns)),
        squeeze=False,
    )
    for goal_index in range(num_goals):
        axis = axes.flat[goal_index]
        goal_rows = [row for row in rows if row["goal_index"] == goal_index]
        distances = np.array([row["distance"] for row in goal_rows])
        costs = np.array([row["latent_mse"] for row in goal_rows])
        axis.scatter(distances[1:], costs[1:], s=22, alpha=0.7)
        axis.scatter(distances[0], costs[0], marker="*", s=100, color="red", label="ball at goal")
        medians = [
            (float(np.median([row["distance"] for row in goal_rows if row["bin_index"] == index])),
             float(np.median([row["latent_mse"] for row in goal_rows if row["bin_index"] == index])))
            for index in range(len(edges) - 1)
            if any(row["bin_index"] == index for row in goal_rows)
        ]
        if medians:
            axis.plot(*zip(*medians), color="black", marker="o", linewidth=1.3, label="bin medians")
        axis.set_title(f"Goal {goal_index}: ({goal_rows[0]['goal_x']:.2f}, {goal_rows[0]['goal_y']:.2f})")
        axis.set_xlabel("Ball-to-goal distance (maze units)")
        axis.set_ylabel("Latent MSE to fixed goal")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    for axis in list(axes.flat)[num_goals:]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(output_dir / "per_goal.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for goal_index in range(num_goals):
        goal_rows = [row for row in rows if row["goal_index"] == goal_index and row["bin_index"] >= 0]
        axis.scatter(
            [row["distance"] for row in goal_rows],
            [row["latent_mse"] for row in goal_rows],
            s=17, alpha=0.35, label=f"goal {goal_index}",
        )
    pooled_medians = [
        (float(np.median([row["distance"] for row in rows if row["bin_index"] == index])),
         float(np.median([row["latent_mse"] for row in rows if row["bin_index"] == index])))
        for index in range(len(edges) - 1)
        if any(row["bin_index"] == index for row in rows)
    ]
    if pooled_medians:
        axis.plot(*zip(*pooled_medians), color="black", marker="o", linewidth=2, label="pooled bin medians")
    axis.scatter([0], [0], marker="*", s=100, color="red", label="ball at goal")
    axis.set_xlabel("Ball-to-goal distance (maze units)")
    axis.set_ylabel("Latent MSE to fixed goal")
    axis.set_title("Top-down I-JEPA distance to rendered goals")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "pooled.png", dpi=160)
    near_costs = [
        row["latent_mse"] for row in rows if row["distance"] <= 0.4
    ]
    axis.set_xlim(-0.01, 0.41)
    axis.set_ylim(-0.001, max(near_costs) * 1.08)
    figure.savefig(output_dir / "pooled_near.png", dpi=160)
    plt.close(figure)


def run_experiment(
    env,
    encoder,
    output_dir,
    num_goals=6,
    base_seed=0,
    grid_resolution=61,
    wall_clearance=0.12,
    per_bin=4,
    batch_size=4,
    device=torch.device("cpu"),
    precision="fp32",
    edges=DEFAULT_EDGES,
):
    if num_goals < 1:
        raise ValueError("num_goals must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env.reset(seed=base_seed)
    wall_rectangles = extract_wall_rectangles(env)
    x_limits, y_limits = maze_limits_from_walls(wall_rectangles)
    valid_positions = generate_valid_grid_positions(
        x_limits, y_limits, wall_rectangles, grid_resolution, wall_clearance
    )
    goal_seeds = select_goal_seeds(env, num_goals, base_seed)
    rng = np.random.default_rng(base_seed)
    rows = []
    for goal_index, goal_seed in enumerate(goal_seeds):
        observation, _ = env.reset(seed=goal_seed)
        hide_goal_marker(env)
        goal = np.asarray(observation["desired_goal"], dtype=np.float32)
        goal_frame = render_target_observation(env)
        positions, bin_ids = sample_positions(goal, valid_positions, edges, per_bin, rng)
        frames = render_positions(env, positions)
        if not np.array_equal(goal_frame, frames[0]):
            raise RuntimeError("goal frame and candidate frame at the goal differ")
        goal_latent = encode_frames(
            encoder, goal_frame[None], device, precision, batch_size
        )[0]
        candidate_latents = encode_frames(
            encoder, frames, device, precision, batch_size
        )
        if goal_latent.ndim != 2 or candidate_latents.shape[1:] != goal_latent.shape:
            raise ValueError("encoder must return matching [patch, feature] latents")
        costs = np.mean((candidate_latents - goal_latent) ** 2, axis=(1, 2))
        distances = np.linalg.norm(positions - goal, axis=1)
        plt.imsave(output_dir / f"goal_{goal_index:02d}.png", goal_frame)
        if len(frames) > 1:
            plt.imsave(output_dir / f"near_{goal_index:02d}.png", frames[1])
        plt.imsave(output_dir / f"far_{goal_index:02d}.png", frames[int(np.argmax(distances))])
        for position, distance, cost, bin_index in zip(positions, distances, costs, bin_ids):
            rows.append({
                "goal_index": goal_index,
                "goal_seed": goal_seed,
                "goal_x": float(goal[0]),
                "goal_y": float(goal[1]),
                "ball_x": float(position[0]),
                "ball_y": float(position[1]),
                "distance": float(distance),
                "bin_index": bin_index,
                "latent_mse": float(cost),
            })
        print(
            f"goal {goal_index + 1}/{num_goals}: position={goal.tolist()}, "
            f"samples={len(positions)}, self_mse={float(costs[0]):.9f}",
            flush=True,
        )

    with (output_dir / "samples.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "num_goals": num_goals,
        "num_samples": len(rows),
        "base_seed": base_seed,
        "goal_seeds": goal_seeds,
        "grid_resolution": grid_resolution,
        "wall_clearance": wall_clearance,
        "per_bin": per_bin,
        "bin_edges": list(edges),
        "camera": "topdown",
        "goal_marker": "hidden",
        "goal_rendering": "green ball at goal, zero velocity",
        "metric": "mean squared error over matching I-JEPA patch features",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_plots(rows, output_dir, edges, num_goals)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-checkpoint", type=Path, default=DEFAULT_ENCODER_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--environment-id", default="PointMaze_UMaze-v3")
    parser.add_argument("--num-goals", type=int, default=6)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--grid-resolution", type=int, default=61)
    parser.add_argument("--wall-clearance", type=float, default=0.12)
    parser.add_argument("--per-bin", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.encoder_checkpoint.is_file():
        raise FileNotFoundError(args.encoder_checkpoint)
    if args.batch_size < 1 or args.per_bin < 1 or args.grid_resolution < 2 or args.wall_clearance < 0:
        raise ValueError("invalid sampling or batch arguments")
    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    print(f"Device: {device}, precision: {precision}", flush=True)
    encoder = load_frozen_ijepa_encoder(args.encoder_checkpoint, device)
    env = create_environment(args.environment_id, max_micro_steps=1, camera="topdown")
    try:
        summary = run_experiment(
            env=env,
            encoder=encoder,
            output_dir=args.output_dir,
            num_goals=args.num_goals,
            base_seed=args.base_seed,
            grid_resolution=args.grid_resolution,
            wall_clearance=args.wall_clearance,
            per_bin=args.per_bin,
            batch_size=args.batch_size,
            device=device,
            precision=precision,
        )
    finally:
        env.close()
    print(f"Saved {summary['num_samples']} comparisons to: {args.output_dir}")


if __name__ == "__main__":
    main()

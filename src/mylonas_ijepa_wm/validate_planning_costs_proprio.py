"""Replay selected imagined plans and compare predicted costs with real outcomes."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.mylonas_ijepa_wm import plan_pointmaze as base
from src.mylonas_ijepa_wm import wm
from src.mylonas_ijepa_wm.embed_frames import preprocess_frames
from src.mylonas_ijepa_wm.planning_cost_rankings_proprio import (
    DEFAULT_OUTPUT_DIR,
    _average_ranks,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--top-per-cost", type=int, default=16)
    parser.add_argument("--random-count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    return parser.parse_args(argv)


def select_candidate_ids(latent_costs, position_costs, top_per_cost, random_count, seed):
    """Take both elite groups and random candidates outside their union."""
    latent_costs = np.asarray(latent_costs)
    position_costs = np.asarray(position_costs)
    if (
        latent_costs.ndim != 1 or position_costs.shape != latent_costs.shape
        or top_per_cost < 1 or top_per_cost > len(latent_costs)
        or random_count < 0
    ):
        raise ValueError("cost arrays and selection counts are invalid")
    latent_elites = set(np.argsort(latent_costs, kind="stable")[:top_per_cost].tolist())
    position_elites = set(np.argsort(position_costs, kind="stable")[:top_per_cost].tolist())
    remaining = np.array(sorted(set(range(len(latent_costs))) - latent_elites - position_elites))
    if random_count > len(remaining):
        raise ValueError("not enough candidates outside the elite groups")
    rng = np.random.default_rng(seed)
    random_ids = set(rng.choice(remaining, size=random_count, replace=False).tolist())
    return {
        "selected_ids": sorted(latent_elites | position_elites | random_ids),
        "latent_elites": latent_elites,
        "position_elites": position_elites,
        "random_ids": random_ids,
    }


def compute_true_latent_cost(actual_latents, goal_latent):
    actual_latents = np.asarray(actual_latents, dtype=np.float32)
    goal_latent = np.asarray(goal_latent, dtype=np.float32)
    if actual_latents.ndim != 3 or actual_latents.shape[1:] != goal_latent.shape:
        raise ValueError("actual latents and goal latent have incompatible shapes")
    return np.square(actual_latents - goal_latent[None]).mean(axis=(1, 2))


def _spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return None
    x_rank = _average_ranks(x)
    y_rank = _average_ranks(y)
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def summarize_case(rows):
    """Keep representation geometry and model prediction checks separate."""
    if len(rows) < 2:
        raise ValueError("at least two candidate outcomes are required")
    values = lambda key: np.array([row[key] for row in rows], dtype=np.float64)
    actual_distance = values("actual_distance")
    true_latent = values("true_latent_cost")
    predicted_latent = values("predicted_latent_cost")
    predicted_position = values("predicted_position_cost")
    return {
        "candidate_count": len(rows),
        "true_latent_vs_actual_distance_rho": _spearman(true_latent, actual_distance),
        "predicted_latent_vs_true_latent_rho": _spearman(predicted_latent, true_latent),
        "predicted_latent_vs_actual_distance_rho": _spearman(predicted_latent, actual_distance),
        "predicted_position_vs_actual_distance_rho": _spearman(predicted_position, actual_distance),
    }


def replay_actions(env, seed, macro_actions, expected_start, expected_goal):
    """Execute every micro action from a fresh, verified reset."""
    observation, _ = env.reset(seed=seed)
    base.hide_goal_marker(env)
    if not np.allclose(observation["achieved_goal"][:2], expected_start, atol=1e-6):
        raise RuntimeError(f"seed {seed} no longer reproduces the saved start")
    if not np.allclose(observation["desired_goal"][:2], expected_goal, atol=1e-6):
        raise RuntimeError(f"seed {seed} no longer reproduces the saved goal")
    initial_frame = base.validate_frame(env.render())
    actions = np.asarray(macro_actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] != 10:
        raise ValueError("macro_actions must have shape [horizon, 10]")
    for action in actions.reshape(-1, 2):
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            raise RuntimeError("candidate replay ended before the planned horizon")
    actual_xy = np.asarray(observation["achieved_goal"][:2], dtype=np.float32)
    terminal_frame = base.validate_frame(env.render()).copy()
    return actual_xy, terminal_frame, initial_frame


def encode_frames(encoder, frames, device, precision, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            image_batch = preprocess_frames(
                torch.from_numpy(frames[start:start + batch_size])
            ).to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=precision == "bf16",
            ):
                tokens = encoder(image_batch)
            outputs.append(tokens.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _default_seeds(cases):
    far = [case for case in cases if case["group"] == "far"]
    if len(far) < 3:
        raise ValueError("automatic selection requires at least three far cases")
    disagreement = sorted(
        far, key=lambda case: case["comparison"]["spearman"]
    )[:2]
    remaining = [case for case in far if case not in disagreement]
    agreement = max(remaining, key=lambda case: case["comparison"]["spearman"])
    return [case["seed"] for case in disagreement + [agreement]]


def _plot(rows_by_seed, summaries, output_path):
    figure, axes = plt.subplots(
        len(rows_by_seed), 3,
        figsize=(15, 4 * len(rows_by_seed)), squeeze=False,
    )
    comparisons = (
        ("actual_distance", "true_latent_cost", "True latent cost vs real distance",
         "true_latent_vs_actual_distance_rho"),
        ("true_latent_cost", "predicted_latent_cost", "Predicted vs true latent cost",
         "predicted_latent_vs_true_latent_rho"),
        ("actual_distance", "predicted_position_cost", "Predicted vs real goal distance",
         "predicted_position_vs_actual_distance_rho"),
    )
    colors = {
        "random": "#999999",
        "latent": "#d95f02",
        "position": "#1b9e77",
        "both": "#7570b3",
    }
    for row_index, (seed, rows) in enumerate(rows_by_seed.items()):
        for column, (x_key, y_key, title, rho_key) in enumerate(comparisons):
            axis = axes[row_index, column]
            for group in colors:
                selected = [item for item in rows if item["selection"] == group]
                if selected:
                    axis.scatter(
                        [item[x_key] for item in selected],
                        [item[y_key] for item in selected],
                        s=35, alpha=0.8, color=colors[group], label=group,
                    )
            rho = summaries[seed][rho_key]
            axis.set_title(f"seed {seed}: {title}\nrank rho={rho:.2f}" if rho is not None else title)
            axis.set_xlabel(x_key.replace("_", " "))
            axis.set_ylabel(y_key.replace("_", " "))
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    if args.random_count < 2:
        raise ValueError("random-count must be at least 2 for the control summary")
    source = json.loads((args.input_dir / "summary.json").read_text())
    seeds = args.seeds if args.seeds is not None else _default_seeds(source["cases"])
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    source_cases = {case["seed"]: case for case in source["cases"]}
    if not set(seeds).issubset(source_cases):
        raise ValueError("all seeds must occur in the source diagnostic")
    output_dir = args.output_dir or args.input_dir / "replay_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "sample_frames"
    frames_dir.mkdir(exist_ok=True)
    encoder_checkpoint = Path(source["configuration"]["encoder_checkpoint"])
    if not encoder_checkpoint.is_file():
        raise FileNotFoundError(encoder_checkpoint)
    device = wm.resolve_device(args.device)
    precision = wm.resolve_precision(args.precision, device)
    encoder = base.load_frozen_ijepa_encoder(encoder_checkpoint, device)
    environment = base.create_environment(
        source["configuration"]["environment_id"],
        max_micro_steps=100,
        camera="topdown",
    )
    rows_by_seed = {}
    summaries = {}
    try:
        for seed in seeds:
            case = source_cases[seed]
            with np.load(args.input_dir / f"seed_{seed:04d}_candidates.npz") as saved:
                actions = saved["actions"]
                predicted_latent = saved["latent_cost"]
                predicted_position = saved["position_cost"]
                predicted_xy = saved["predicted_xy"]
            selection = select_candidate_ids(
                predicted_latent, predicted_position,
                args.top_per_cost, args.random_count, seed=seed + 1729,
            )
            goal_frame = np.asarray(Image.open(
                args.input_dir / "frames" / f"seed_{seed:04d}_goal.png"
            ).convert("RGB"))
            environment.reset(seed=seed)
            base.hide_goal_marker(environment)
            fresh_goal_frame = base.render_target_observation(environment)
            if not np.array_equal(goal_frame, fresh_goal_frame):
                raise RuntimeError(f"seed {seed} goal frame differs from the source diagnostic")
            expected_current = np.asarray(Image.open(
                args.input_dir / "frames" / f"seed_{seed:04d}_current.png"
            ).convert("RGB"))
            selected_ids = selection["selected_ids"]
            actual_positions = []
            terminal_frames = []
            for index in selected_ids:
                actual_xy, frame, initial_frame = replay_actions(
                    environment, seed, actions[index],
                    case["start_xy"], case["goal_xy"],
                )
                if not np.array_equal(initial_frame, expected_current):
                    raise RuntimeError(f"seed {seed} camera frame differs from the source diagnostic")
                actual_positions.append(actual_xy)
                terminal_frames.append(frame)
            latents = encode_frames(
                encoder,
                np.stack([goal_frame, *terminal_frames]),
                device, precision, args.batch_size,
            )
            true_latent_cost = compute_true_latent_cost(latents[1:], latents[0])
            actual_positions = np.stack(actual_positions)
            actual_distance = np.linalg.norm(
                actual_positions - np.asarray(case["goal_xy"]), axis=1,
            )
            rows = []
            for offset, index in enumerate(selected_ids):
                in_latent = index in selection["latent_elites"]
                in_position = index in selection["position_elites"]
                group = (
                    "both" if in_latent and in_position else
                    "latent" if in_latent else
                    "position" if in_position else "random"
                )
                rows.append({
                    "seed": seed, "candidate": index, "selection": group,
                    "predicted_latent_cost": float(predicted_latent[index]),
                    "predicted_position_cost": float(predicted_position[index]),
                    "predicted_x": float(predicted_xy[index, 0]),
                    "predicted_y": float(predicted_xy[index, 1]),
                    "true_latent_cost": float(true_latent_cost[offset]),
                    "actual_distance": float(actual_distance[offset]),
                    "actual_x": float(actual_positions[offset, 0]),
                    "actual_y": float(actual_positions[offset, 1]),
                })
            rows_by_seed[seed] = rows
            metrics = summarize_case(rows)
            metrics["random_only"] = summarize_case([
                row for row in rows if row["selection"] == "random"
            ])
            metrics["median_actual_distance_by_selection"] = {
                group: float(np.median([
                    row["actual_distance"] for row in rows
                    if row["selection"] == group
                ]))
                for group in ("latent", "position", "both", "random")
                if any(row["selection"] == group for row in rows)
            }
            metrics["mean_position_prediction_error"] = float(np.mean(
                np.linalg.norm(
                    predicted_xy[selected_ids] - actual_positions, axis=1,
                )
            ))
            summaries[seed] = metrics
            for label, index in (
                ("latent_best", int(np.argmin(predicted_latent))),
                ("position_best", int(np.argmin(predicted_position))),
                ("random", min(selection["random_ids"])),
            ):
                offset = selected_ids.index(index)
                Image.fromarray(terminal_frames[offset]).save(
                    frames_dir / f"seed_{seed:04d}_{label}.png"
                )
            print(
                f"seed {seed}: {len(rows)} replays, "
                f"true latent vs distance rho={metrics['true_latent_vs_actual_distance_rho']:.3f}, "
                f"predicted vs true latent rho={metrics['predicted_latent_vs_true_latent_rho']:.3f}, "
                f"predicted position vs real distance rho={metrics['predicted_position_vs_actual_distance_rho']:.3f}",
                flush=True,
            )
    finally:
        environment.close()

    all_rows = [row for rows in rows_by_seed.values() for row in rows]
    with (output_dir / "replayed_candidates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    _plot(rows_by_seed, summaries, output_dir / "replay_cost_diagnostics.png")
    summary = {
        "source_diagnostic": str(args.input_dir),
        "encoder_checkpoint": str(encoder_checkpoint),
        "seeds": seeds,
        "top_per_cost": args.top_per_cost,
        "random_count": args.random_count,
        "sampling_note": "Correlations across the selected candidates are selection-biased. random_only samples outside the two elite groups, so it also does not represent the full candidate pool.",
        "device": str(device),
        "precision": precision,
        "cases": {
            str(seed): {
                "start_xy": source_cases[seed]["start_xy"],
                "goal_xy": source_cases[seed]["goal_xy"],
                "initial_distance": source_cases[seed]["initial_distance"],
                **summaries[seed],
            }
            for seed in seeds
        },
    }
    base._write_json_atomically(summary, output_dir / "summary.json")
    print(f"Saved replay validation to {output_dir}")


if __name__ == "__main__":
    main()

"""Compare visual and predicted-position goal costs on identical imagined plans."""

import argparse
import csv
import gc
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.mylonas_ijepa_wm import plan_pointmaze as base
from src.mylonas_ijepa_wm import plan_pointmaze_proprio as planner
from src.mylonas_ijepa_wm import wm, wm_proprio
from src.mylonas_ijepa_wm.planning import compute_terminal_cost


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    MODULE_DIR / "proprio" / "world_model_runs" / "full_default_v1_k3" / "best.pt"
)
DEFAULT_OUTPUT_DIR = MODULE_DIR / "debug" / "planning_cost_rankings_proprio"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world-model-checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument(
        "--encoder-checkpoint", type=Path,
        default=base.DEFAULT_ENCODER_CHECKPOINT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--environment-id", default="PointMaze_UMaze-v3")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--seed-pool", type=int, default=20)
    parser.add_argument("--num-far", type=int, default=5)
    parser.add_argument("--num-near", type=int, default=2)
    parser.add_argument("--planning-horizon", type=int, default=3)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--num-elites", type=int, default=32)
    parser.add_argument("--candidate-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    return parser.parse_args(argv)


def score_candidate_sequences(
    model,
    latent_history,
    state_history,
    action_history,
    candidate_actions,
    target_latent,
    goal_position,
    state_mean,
    state_std,
    device,
    candidate_batch_size,
    action_mean=None,
    action_std=None,
    precision="fp32",
):
    """Score both costs from the same rollouts, without retaining visual batches."""
    latent_costs = []
    position_costs = []
    predicted_positions = []
    mean_xy = torch.as_tensor(state_mean, device=device, dtype=torch.float32)[:2]
    std_xy = torch.as_tensor(state_std, device=device, dtype=torch.float32)[:2]
    if candidate_batch_size < 1 or len(candidate_actions) == 0:
        raise ValueError("candidate_batch_size and candidate count must be positive")

    for start in range(0, len(candidate_actions), candidate_batch_size):
        chunk = candidate_actions[start:start + candidate_batch_size]
        predicted_visual, predicted_states = planner.rollout_action_sequences(
            model=model,
            latent_history=latent_history.unsqueeze(0),
            state_history=state_history.unsqueeze(0),
            action_history=action_history,
            candidate_actions=chunk,
            device=device,
            chunk_size=candidate_batch_size,
            action_mean=action_mean,
            action_std=action_std,
            precision=precision,
        )
        latent_costs.append(compute_terminal_cost(
            predicted_latents=predicted_visual,
            target_latent=target_latent,
            mode="latent",
        ).detach().cpu().numpy())
        position_costs.append(planner.compute_terminal_position_cost(
            predicted_states=predicted_states,
            goal_position=goal_position,
            state_mean=state_mean,
            state_std=state_std,
        ).detach().cpu().numpy())
        predicted_positions.append((
            predicted_states[:, -1, :2].float() * std_xy + mean_xy
        ).detach().cpu().numpy())

    result = {
        "latent_cost": np.concatenate(latent_costs),
        "position_cost": np.concatenate(position_costs),
        "predicted_xy": np.concatenate(predicted_positions),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("world model produced nonfinite candidate predictions")
    return result


def _average_ranks(values):
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def compare_rankings(latent_costs, position_costs, num_elites):
    """Summarize whether CEM would select the same candidates under both costs."""
    latent_costs = np.asarray(latent_costs, dtype=np.float64)
    position_costs = np.asarray(position_costs, dtype=np.float64)
    if (
        latent_costs.ndim != 1 or position_costs.shape != latent_costs.shape
        or len(latent_costs) < 2 or not 1 <= num_elites <= len(latent_costs)
        or not np.isfinite(latent_costs).all()
        or not np.isfinite(position_costs).all()
    ):
        raise ValueError("costs must be finite aligned vectors with valid num_elites")

    latent_order = np.argsort(latent_costs, kind="stable")
    position_order = np.argsort(position_costs, kind="stable")
    latent_elites = latent_order[:num_elites]
    position_elites = position_order[:num_elites]
    latent_ranks = _average_ranks(latent_costs)
    position_ranks = _average_ranks(position_costs)
    if np.std(latent_ranks) == 0 or np.std(position_ranks) == 0:
        spearman = None
    else:
        spearman = float(np.corrcoef(latent_ranks, position_ranks)[0, 1])
    overlap = len(set(latent_elites) & set(position_elites))
    return {
        "spearman": spearman,
        "elite_overlap_count": overlap,
        "elite_overlap_fraction": overlap / num_elites,
        "random_overlap_expected": num_elites * num_elites / len(latent_costs),
        "latent_best_position_distance": float(position_costs[latent_order[0]]),
        "position_best_distance": float(position_costs[position_order[0]]),
        "latent_elite_median_position_distance": float(np.median(position_costs[latent_elites])),
        "position_elite_median_position_distance": float(np.median(position_costs[position_elites])),
    }


def select_cases(env, base_seed, seed_pool, num_far, num_near):
    if seed_pool < 2 or num_far < 1 or num_near < 0 or num_far + num_near > seed_pool:
        raise ValueError("case counts must fit within a seed pool of at least two")
    pool = []
    for seed in range(base_seed, base_seed + seed_pool):
        observation, _ = env.reset(seed=seed)
        pool.append({"seed": seed, "initial_distance": base._goal_distance(observation)})
    far = sorted(pool, key=lambda item: (-item["initial_distance"], item["seed"]))[:num_far]
    remaining = [item for item in pool if item not in far]
    near = sorted(remaining, key=lambda item: (item["initial_distance"], item["seed"]))[:num_near]
    return pool, [dict(item, group="far") for item in far] + [
        dict(item, group="near") for item in near
    ]


def draw_initial_cem_candidates(seed, population, horizon, low, high):
    """Match the first sampling round of planning.cem_optimize."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    candidates = torch.randn(
        population, horizon, len(low), generator=generator
    )
    return candidates.clamp(
        min=torch.as_tensor(low, dtype=torch.float32),
        max=torch.as_tensor(high, dtype=torch.float32),
    )


def save_scatter_plot(case_results, num_elites, output_path):
    columns = min(3, len(case_results))
    rows = (len(case_results) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False)
    for axis, case in zip(axes.flat, case_results):
        scores = case["scores"]
        position = scores["position_cost"]
        latent = scores["latent_cost"]
        latent_elite = set(np.argsort(latent, kind="stable")[:num_elites])
        position_elite = set(np.argsort(position, kind="stable")[:num_elites])
        groups = (
            (set(range(len(latent))) - latent_elite - position_elite, "other", "#a4a4a4"),
            (latent_elite - position_elite, "latent elite only", "#d95f02"),
            (position_elite - latent_elite, "position elite only", "#1b9e77"),
            (latent_elite & position_elite, "both elites", "#7570b3"),
        )
        for indices, label, color in groups:
            if indices:
                selected = np.array(sorted(indices))
                axis.scatter(position[selected], latent[selected], s=22, alpha=0.75, color=color, label=label)
        correlation = case["comparison"]["spearman"]
        rho = "undefined" if correlation is None else f"{correlation:.2f}"
        axis.set_title(
            f"{case['group']} seed {case['seed']} | start d={case['initial_distance']:.2f}\n"
            f"rank rho={rho}, elite overlap={case['comparison']['elite_overlap_count']}/{num_elites}"
        )
        axis.set_xlabel("WM-predicted terminal distance to goal")
        axis.set_ylabel("WM-predicted terminal latent MSE")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    for axis in list(axes.flat)[len(case_results):]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    for path in (args.world_model_checkpoint, args.encoder_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.population < 2 or not 1 <= args.num_elites <= args.population:
        raise ValueError("population and num_elites must define a valid elite set")
    if args.planning_horizon < 1 or args.candidate_batch_size < 1:
        raise ValueError("planning_horizon and candidate_batch_size must be positive")
    device = wm.resolve_device(args.device)
    precision = wm.resolve_precision(args.precision, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    environment = base.create_environment(
        args.environment_id, max_micro_steps=100, camera="topdown"
    )
    try:
        pool, selected = select_cases(
            environment, args.base_seed, args.seed_pool, args.num_far, args.num_near
        )
        encoder = base.load_frozen_ijepa_encoder(args.encoder_checkpoint, device)
        for case in selected:
            observation, _ = environment.reset(seed=case["seed"])
            base.hide_goal_marker(environment)
            current_frame = base.validate_frame(environment.render())
            goal_frame = base.render_target_observation(environment)
            Image.fromarray(current_frame).save(frames_dir / f"seed_{case['seed']:04d}_current.png")
            Image.fromarray(goal_frame).save(frames_dir / f"seed_{case['seed']:04d}_goal.png")
            case["start_xy"] = np.asarray(observation["achieved_goal"], dtype=np.float32).tolist()
            case["goal_xy"] = np.asarray(observation["desired_goal"], dtype=np.float32).tolist()
            case["state"] = torch.as_tensor(observation["observation"], dtype=torch.float32)
            case["current_latent"] = base.encode_frame(
                encoder, current_frame, device, precision
            ).float().cpu()
            case["goal_latent"] = base.encode_frame(
                encoder, goal_frame, device, precision
            ).float().cpu()
            print(f"Encoded {case['group']} seed {case['seed']}: start distance {case['initial_distance']:.3f}", flush=True)
        del encoder
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

        model, metadata, epoch = wm_proprio.load_model_and_metadata(
            args.world_model_checkpoint, device
        )
        action_low = np.tile(np.asarray(environment.action_space.low), 5)
        action_high = np.tile(np.asarray(environment.action_space.high), 5)
        state_mean = metadata["state_mean"].to(device)
        state_std = metadata["state_std"].to(device)
        case_results = []
        candidate_rows = []
        for case in selected:
            candidates = draw_initial_cem_candidates(
                case["seed"], args.population, args.planning_horizon,
                action_low, action_high,
            )
            current_latent = case.pop("current_latent").to(device)
            goal_latent = case.pop("goal_latent").to(device)
            state = case.pop("state").to(device)
            latent_history = current_latent.unsqueeze(0).repeat(model.num_hist, 1, 1)
            state_history = ((state - state_mean) / state_std).unsqueeze(0).repeat(model.num_hist, 1)
            action_history = torch.zeros(model.num_hist - 1, len(action_low), device=device)
            scores = score_candidate_sequences(
                model=model,
                latent_history=latent_history,
                state_history=state_history,
                action_history=action_history,
                candidate_actions=candidates,
                target_latent=goal_latent,
                goal_position=torch.tensor(case["goal_xy"], device=device),
                state_mean=state_mean,
                state_std=state_std,
                device=device,
                candidate_batch_size=args.candidate_batch_size,
                action_mean=metadata.get("action_mean"),
                action_std=metadata.get("action_std"),
                precision=precision,
            )
            comparison = compare_rankings(
                scores["latent_cost"], scores["position_cost"], args.num_elites
            )
            case["comparison"] = comparison
            case["scores"] = scores
            case_results.append(case)
            latent_elites = set(np.argsort(scores["latent_cost"], kind="stable")[:args.num_elites])
            position_elites = set(np.argsort(scores["position_cost"], kind="stable")[:args.num_elites])
            for index in range(args.population):
                candidate_rows.append({
                    "seed": case["seed"], "group": case["group"], "candidate": index,
                    "latent_cost": float(scores["latent_cost"][index]),
                    "position_cost": float(scores["position_cost"][index]),
                    "predicted_x": float(scores["predicted_xy"][index, 0]),
                    "predicted_y": float(scores["predicted_xy"][index, 1]),
                    "latent_elite": index in latent_elites,
                    "position_elite": index in position_elites,
                })
            np.savez_compressed(
                args.output_dir / f"seed_{case['seed']:04d}_candidates.npz",
                actions=candidates.numpy(), **scores,
            )
            rho = comparison["spearman"]
            print(
                f"Scored seed {case['seed']}: rho={rho if rho is not None else 'undefined'}, "
                f"elite overlap={comparison['elite_overlap_count']}/{args.num_elites}",
                flush=True,
            )
    finally:
        environment.close()

    with (args.output_dir / "seed_pool.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", "initial_distance"])
        writer.writeheader()
        writer.writerows(pool)
    with (args.output_dir / "candidates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    save_scatter_plot(case_results, args.num_elites, args.output_dir / "rankings.png")
    summary = {
        "configuration": {
            "world_model_checkpoint": str(args.world_model_checkpoint),
            "world_model_epoch": epoch,
            "encoder_checkpoint": str(args.encoder_checkpoint),
            "environment_id": args.environment_id,
            "base_seed": args.base_seed,
            "seed_pool": args.seed_pool,
            "num_far": args.num_far,
            "num_near": args.num_near,
            "planning_horizon": args.planning_horizon,
            "population": args.population,
            "num_elites": args.num_elites,
            "candidate_batch_size": args.candidate_batch_size,
            "candidate_sampler": "first CEM iteration: N(0,1), clipped to action bounds",
            "device": str(device),
            "precision": precision,
        },
        "cases": [
            {key: value for key, value in case.items() if key != "scores"}
            for case in case_results
        ],
    }
    base._write_json_atomically(summary, args.output_dir / "summary.json")
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()

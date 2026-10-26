"""Replay Wall CEM candidates to compare predicted and real visual goal costs."""

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.mylonas_ijepa_wm import plan_pointmaze as visual
from src.mylonas_ijepa_wm import plan_pointmaze_proprio as planner
from src.mylonas_ijepa_wm import wm, wm_proprio
from src.mylonas_ijepa_wm.environment_adapters import create_adapter
from src.mylonas_ijepa_wm.environment_configs import load_environment_config
from src.mylonas_ijepa_wm.validate_planning_costs_proprio import (
    _spearman,
    compute_true_latent_cost,
    encode_frames,
)


def select_candidates(rounds, random_count, elite_count, seed):
    """Keep uniform first-round controls and the best final-round plans."""
    if not rounds or min(random_count, elite_count) < 1:
        raise ValueError("CEM rounds and sample counts must be positive")
    first_actions, first_costs = rounds[0]
    final_actions, final_costs = rounds[-1]
    if random_count > len(first_costs) or elite_count > len(final_costs):
        raise ValueError("requested more candidates than the CEM population")
    rng = np.random.default_rng(seed)
    initial_ids = sorted(rng.choice(
        len(first_costs), size=random_count, replace=False
    ).tolist())
    final_ids = np.argsort(final_costs.numpy(), kind="stable")[:elite_count].tolist()
    selected = []
    for source, round_index, ids, actions, costs in (
        ("initial_random", 0, initial_ids, first_actions, first_costs),
        ("final_elite", len(rounds) - 1, final_ids, final_actions, final_costs),
    ):
        for candidate_id in ids:
            selected.append({
                "source": source,
                "round": round_index,
                "candidate_id": int(candidate_id),
                "actions": actions[candidate_id].numpy().astype(np.float32),
                "predicted_cost": float(costs[candidate_id]),
            })
    return selected


def replay_candidates(adapter, seed, candidates, actions_per_model_step):
    """Run each candidate from the same reset and return its real endpoint."""
    if actions_per_model_step < 1 or not candidates:
        raise ValueError("replay needs candidates and a positive action stride")
    outcomes = []
    expected_start = expected_goal = None
    for candidate in candidates:
        initial = adapter.reset(seed)
        if expected_start is None:
            expected_start = np.asarray(initial.state).copy()
            expected_goal = np.asarray(adapter.goal).copy()
        elif not (np.allclose(initial.state, expected_start)
                  and np.allclose(adapter.goal, expected_goal)):
            raise RuntimeError("Wall reset changed the start or goal")
        actions = np.asarray(candidate["actions"], dtype=np.float32)
        if (actions.ndim != 2 or len(actions) < 1
                or actions.shape[1] != actions_per_model_step * 2):
            raise ValueError("candidate actions do not match the Wall macro stride")
        reached_goal = False
        for action in actions.reshape(-1, 2):
            outcome = adapter.step(action)
            reached_goal |= bool(outcome.success)
        outcomes.append({
            "state": np.asarray(outcome.state, dtype=np.float32).copy(),
            "frame": adapter.render().copy(),
            "reached_goal": reached_goal,
        })
    return outcomes


def summarize_rankings(rows):
    """Separate real goal geometry from world-model ranking error."""
    if len(rows) < 2:
        raise ValueError("at least two candidates are required")
    predicted = np.asarray([row["predicted_cost"] for row in rows])
    true = np.asarray([row["true_cost"] for row in rows])
    distances = np.asarray([row["actual_distance"] for row in rows])
    predicted_best = int(np.argmin(predicted))
    return {
        "candidate_count": len(rows),
        "predicted_vs_true_rho": _spearman(predicted, true),
        "true_vs_distance_rho": _spearman(true, distances),
        "predicted_vs_distance_rho": _spearman(predicted, distances),
        "predicted_best_true_cost": float(true[predicted_best]),
        "oracle_best_true_cost": float(true.min()),
        "predicted_best_actual_distance": float(distances[predicted_best]),
        "oracle_best_actual_distance": float(distances.min()),
        "top_ten_overlap": len(
            set(np.argsort(predicted)[:min(10, len(rows))])
            & set(np.argsort(true)[:min(10, len(rows))])
        ),
    }


def validate_multi_wall_checkpoint(metadata, wall_x, door_y):
    """Require provenance for the trained multi-layout model and a seen layout."""
    splits = metadata.get("layout_splits")
    if (metadata.get("environment") != "wall"
            or not metadata.get("experiment_manifest_sha256")
            or not isinstance(splits, dict)
            or not splits.get("train")):
        raise ValueError("checkpoint is not a multi-wall Wall checkpoint")
    if [wall_x, door_y] not in splits["train"]:
        raise ValueError("requested wall and door are not a checkpoint training layout")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-model-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wall-x", type=int, default=32)
    parser.add_argument("--door-y", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=10000)
    parser.add_argument("--seed-pool", type=int, default=100)
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--min-initial-distance", type=float, default=20.0)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--population", type=int, default=300)
    parser.add_argument("--num-elites", type=int, default=30)
    parser.add_argument("--cem-iterations", type=int, default=10)
    parser.add_argument("--initial-samples", type=int, default=32)
    parser.add_argument("--final-elites", type=int, default=30)
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument("--encoder-batch-size", type=int, default=4)
    parser.add_argument("--max-action-length", type=float, default=3.5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for path in (args.world_model_checkpoint, args.encoder_checkpoint, args.env_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (min(args.seed_pool, args.num_cases, args.population, args.num_elites,
            args.cem_iterations, args.initial_samples, args.final_elites,
            args.candidate_batch_size, args.encoder_batch_size, *args.horizons) < 1
            or args.num_cases > args.seed_pool or args.num_elites > args.population
            or args.initial_samples > args.population
            or args.final_elites > args.num_elites
            or not np.isfinite(args.max_action_length) or args.max_action_length <= 0):
        raise ValueError("invalid case, CEM, batch, horizon, or action-length setting")
    if len(set(args.horizons)) != len(args.horizons):
        raise ValueError("horizons must be unique")

    config = load_environment_config("wall", args.env_config)
    config = replace(
        config, wall_location=args.wall_x, door_location=args.door_y,
        wall_layouts=[[args.wall_x, args.door_y]],
    )
    device = wm.resolve_device(args.device)
    precision = wm.resolve_precision(args.precision, device)
    model, metadata, epoch = wm_proprio.load_model_and_metadata(
        args.world_model_checkpoint, device
    )
    validate_multi_wall_checkpoint(metadata, args.wall_x, args.door_y)
    shape = metadata["model_configuration"]
    if shape["action_dim"] != config.macro_action_dim or shape["proprio_dim"] != 2:
        raise ValueError("checkpoint action or state shape does not match Wall")
    encoder = visual.load_frozen_ijepa_encoder(args.encoder_checkpoint, device)
    adapter = create_adapter(config, max(args.horizons) * config.actions_per_model_step)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    all_rows = []
    try:
        for seed in range(args.base_seed, args.base_seed + args.seed_pool):
            initial = adapter.reset(seed)
            start = np.asarray(initial.state[:2], dtype=np.float32)
            goal = np.asarray(adapter.goal[:2], dtype=np.float32)
            if ((start[0] - args.wall_x) * (goal[0] - args.wall_x) < 0
                    and np.linalg.norm(start - goal) >= args.min_initial_distance):
                cases.append((seed, start.copy(), goal.copy()))
            if len(cases) == args.num_cases:
                break
        if len(cases) < args.num_cases:
            raise ValueError("seed pool did not contain enough far, opposite-side goals")
        print("Selected cases:", [seed for seed, _, _ in cases], flush=True)

        state_mean = metadata["state_mean"].to(device=device, dtype=torch.float32)
        state_std = metadata["state_std"].to(device=device, dtype=torch.float32)
        low = torch.full((config.macro_action_dim,), -torch.inf, device=device)
        high = torch.full((config.macro_action_dim,), torch.inf, device=device)
        summaries = []
        for seed, start, goal in cases:
            for horizon in args.horizons:
                initial = adapter.reset(seed)
                goal_frame = adapter.render_goal().copy()
                current_latent = visual.encode_frame(
                    encoder, initial.frame, device, precision
                )
                goal_latent = visual.encode_frame(
                    encoder, goal_frame, device, precision
                )
                history = current_latent.unsqueeze(0).repeat(model.num_hist, 1, 1)
                state = torch.as_tensor(initial.state, device=device, dtype=torch.float32)
                state_history = ((state - state_mean) / state_std).unsqueeze(0).repeat(
                    model.num_hist, 1
                )
                action_history = torch.zeros(
                    model.num_hist - 1, config.macro_action_dim, device=device
                )
                rounds = []

                def observe(_, actions, costs):
                    rounds.append((actions.cpu(), costs.cpu()))

                planner.plan_action(
                    model=model, latent_history=history,
                    state_history=state_history, action_history=action_history,
                    target_latent=goal_latent, action_low=low, action_high=high,
                    device=device, planning_horizon=horizon,
                    population=args.population, num_elites=args.num_elites,
                    num_iterations=args.cem_iterations,
                    candidate_batch_size=args.candidate_batch_size,
                    action_mean=metadata.get("action_mean"),
                    action_std=metadata.get("action_std"), precision=precision,
                    generator=visual._create_generator(device, seed + 1000 * horizon),
                    cost="latent", max_action_length=args.max_action_length,
                    micro_action_dim=config.micro_action_dim,
                    candidate_observer=observe,
                )
                selected = select_candidates(
                    rounds, args.initial_samples, args.final_elites,
                    seed + 1000 * horizon,
                )
                outcomes = replay_candidates(
                    adapter, seed, selected, config.actions_per_model_step
                )
                frames = np.stack([goal_frame, *(item["frame"] for item in outcomes)])
                latents = encode_frames(
                    encoder, frames, device, precision, args.encoder_batch_size
                )
                true_costs = compute_true_latent_cost(latents[1:], latents[0])
                rows = []
                for item, outcome, true_cost in zip(selected, outcomes, true_costs):
                    end = outcome["state"][:2]
                    row = {
                        "seed": seed, "wall_x": args.wall_x, "door_y": args.door_y,
                        "horizon": horizon, "source": item["source"],
                        "round": item["round"], "candidate_id": item["candidate_id"],
                        "predicted_cost": item["predicted_cost"],
                        "true_cost": float(true_cost),
                        "actual_distance": float(np.linalg.norm(end - goal)),
                        "start_x": float(start[0]), "start_y": float(start[1]),
                        "goal_x": float(goal[0]), "goal_y": float(goal[1]),
                        "end_x": float(end[0]), "end_y": float(end[1]),
                        "reached_goal": bool(outcome["reached_goal"]),
                        "crossed_wall": bool((start[0] - args.wall_x) * (end[0] - args.wall_x) < 0),
                    }
                    rows.append(row)
                    all_rows.append(row)

                key = f"seed_{seed}_h{horizon}"
                np.savez_compressed(
                    args.output_dir / f"{key}_actions.npz",
                    actions=np.stack([item["actions"] for item in selected]),
                    source=np.asarray([item["source"] for item in selected]),
                    predicted_cost=np.asarray([item["predicted_cost"] for item in selected]),
                )
                groups = {
                    "all": summarize_rankings(rows),
                    "initial_random": summarize_rankings(
                        [row for row in rows if row["source"] == "initial_random"]
                    ),
                    "final_elite": summarize_rankings(
                        [row for row in rows if row["source"] == "final_elite"]
                    ),
                }
                summaries.append({
                    "seed": seed, "horizon": horizon,
                    "start": start.tolist(), "goal": goal.tolist(),
                    "rankings": groups,
                })
                figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
                for source, color in (("initial_random", "#888888"),
                                      ("final_elite", "#cf5b28")):
                    chosen = [row for row in rows if row["source"] == source]
                    axes[0].scatter(
                        [row["true_cost"] for row in chosen],
                        [row["predicted_cost"] for row in chosen],
                        label=source, color=color, alpha=0.75,
                    )
                    axes[1].scatter(
                        [row["actual_distance"] for row in chosen],
                        [row["true_cost"] for row in chosen],
                        label=source, color=color, alpha=0.75,
                    )
                axes[0].set(xlabel="Real endpoint latent MSE",
                            ylabel="WM-predicted endpoint latent MSE",
                            title="Does the WM preserve candidate rankings?")
                axes[1].set(xlabel="Real endpoint ball-to-goal distance",
                            ylabel="Real endpoint latent MSE",
                            title="Does the encoder cost track progress?")
                for axis in axes:
                    axis.grid(alpha=0.2)
                    axis.legend()
                figure.suptitle(f"Wall ({args.wall_x}, {args.door_y}), seed {seed}, horizon {horizon}")
                figure.tight_layout()
                figure.savefig(args.output_dir / f"{key}.png", dpi=160)
                plt.close(figure)
                print(
                    f"{key}: predicted/real rho={groups['all']['predicted_vs_true_rho']}, "
                    f"true/distance rho={groups['all']['true_vs_distance_rho']}",
                    flush=True,
                )
    finally:
        adapter.close()

    with (args.output_dir / "candidates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    visual._write_json_atomically({
        "configuration": {
            "world_model_checkpoint": str(args.world_model_checkpoint),
            "encoder_checkpoint": str(args.encoder_checkpoint),
            "checkpoint_epoch": epoch,
            "environment": config.portable_metadata(),
            "population": args.population,
            "num_elites": args.num_elites,
            "cem_iterations": args.cem_iterations,
            "initial_samples": args.initial_samples,
            "final_elites": args.final_elites,
            "max_action_length": args.max_action_length,
            "seed_selection": "first far, opposite-side cases in seed pool",
        },
        "cases": summaries,
    }, args.output_dir / "summary.json")
    print("Saved candidate replay to:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()

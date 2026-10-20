"""Compare I-JEPA and DINOv2 goal costs on identical real PointMaze outcomes."""

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
from src.mylonas_ijepa_wm.planning_cost_rankings_proprio import DEFAULT_OUTPUT_DIR
from src.mylonas_ijepa_wm.validate_planning_costs_proprio import (
    _spearman,
    compute_true_latent_cost,
    encode_frames,
    replay_actions,
)


DEFAULT_DINOV2_CHECKPOINT = (
    Path(__file__).resolve().parent.parent
    / "checkpoints" / "dinov2_vits14_pretrain.pth"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dinov2-checkpoint", type=Path, default=DEFAULT_DINOV2_CHECKPOINT)
    parser.add_argument("--ijepa-checkpoint", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 18, 19])
    parser.add_argument("--candidates-per-seed", type=int, default=64)
    parser.add_argument("--sampling-seed", type=int, default=20261019)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    return parser.parse_args(argv)


def sample_candidate_ids(population, count, seed):
    """Sample uniformly from actions drawn before either planning cost was computed."""
    if population < 2 or not 2 <= count <= population:
        raise ValueError("count must be between 2 and the candidate population")
    return sorted(np.random.default_rng(seed).choice(
        population, size=count, replace=False,
    ).tolist())


def summarize_geometry(rows):
    if len(rows) < 2:
        raise ValueError("at least two real outcomes are required")
    actual = [row["actual_distance"] for row in rows]
    return {
        "candidate_count": len(rows),
        "actual_distance_min": float(min(actual)),
        "actual_distance_max": float(max(actual)),
        "ijepa_distance_spearman": _spearman(
            [row["ijepa_mse"] for row in rows], actual,
        ),
        "dinov2_distance_spearman": _spearman(
            [row["dinov2_mse"] for row in rows], actual,
        ),
    }


class DinoPatchEncoder(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, images):
        return self.backbone.forward_features(images)["x_norm_patchtokens"]


def load_dinov2_encoder(checkpoint_path, device):
    backbone = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14",
        pretrained=False, trust_repo=True,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state, strict=True)
    encoder = DinoPatchEncoder(backbone).to(device).eval()
    encoder.requires_grad_(False)
    return encoder


def plot_geometry(rows_by_seed, summaries, path):
    figure, axes = plt.subplots(
        len(rows_by_seed), 2, figsize=(11, 3.7 * len(rows_by_seed)),
        squeeze=False,
    )
    for row_index, (seed, rows) in enumerate(rows_by_seed.items()):
        for column, (cost_key, title, color) in enumerate((
            ("ijepa_mse", "I-JEPA", "#d95f02"),
            ("dinov2_mse", "DINOv2 ViT-S/14", "#1b9e77"),
        )):
            axis = axes[row_index, column]
            axis.scatter(
                [row["actual_distance"] for row in rows],
                [row[cost_key] for row in rows],
                s=28, alpha=0.7, color=color,
            )
            rho = summaries[seed][f"{'ijepa' if column == 0 else 'dinov2'}_distance_spearman"]
            axis.set_title(f"Seed {seed}: {title} (Spearman rho={rho:.2f})")
            axis.set_xlabel("Actual terminal distance to goal")
            axis.set_ylabel("Real frame-to-goal patch MSE")
            axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("seeds must be unique")
    source = json.loads((args.input_dir / "summary.json").read_text())
    source_cases = {case["seed"]: case for case in source["cases"]}
    if not set(args.seeds).issubset(source_cases):
        raise ValueError("all seeds must occur in the saved candidate diagnostic")
    ijepa_checkpoint = args.ijepa_checkpoint or Path(
        source["configuration"]["encoder_checkpoint"]
    )
    for checkpoint in (ijepa_checkpoint, args.dinov2_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    output_dir = args.output_dir or args.input_dir / "encoder_goal_geometry"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = wm.resolve_device(args.device)
    precision = wm.resolve_precision(args.precision, device)
    ijepa = base.load_frozen_ijepa_encoder(ijepa_checkpoint, device)
    dinov2 = load_dinov2_encoder(args.dinov2_checkpoint, device)
    environment = base.create_environment(
        source["configuration"]["environment_id"],
        max_micro_steps=100, camera="topdown",
    )
    rows_by_seed = {}
    summaries = {}
    try:
        for seed in args.seeds:
            case = source_cases[seed]
            with np.load(args.input_dir / f"seed_{seed:04d}_candidates.npz") as saved:
                actions = saved["actions"]
            candidate_ids = sample_candidate_ids(
                len(actions), args.candidates_per_seed,
                seed=args.sampling_seed + seed,
            )
            goal_frame = np.asarray(Image.open(
                args.input_dir / "frames" / f"seed_{seed:04d}_goal.png"
            ).convert("RGB"))
            current_frame = np.asarray(Image.open(
                args.input_dir / "frames" / f"seed_{seed:04d}_current.png"
            ).convert("RGB"))
            environment.reset(seed=seed)
            base.hide_goal_marker(environment)
            if not np.array_equal(goal_frame, base.render_target_observation(environment)):
                raise RuntimeError(f"seed {seed} goal frame changed")
            positions = []
            terminal_frames = []
            for candidate in candidate_ids:
                actual_xy, terminal_frame, initial_frame = replay_actions(
                    environment, seed, actions[candidate],
                    case["start_xy"], case["goal_xy"],
                )
                if not np.array_equal(initial_frame, current_frame):
                    raise RuntimeError(f"seed {seed} start frame changed")
                positions.append(actual_xy)
                terminal_frames.append(terminal_frame)
            frames = np.stack([goal_frame, *terminal_frames])
            ijepa_latents = encode_frames(
                ijepa, frames, device, precision, args.batch_size,
            )
            ijepa_cost = compute_true_latent_cost(
                ijepa_latents[1:], ijepa_latents[0],
            )
            del ijepa_latents
            dinov2_latents = encode_frames(
                dinov2, frames, device, precision, args.batch_size,
            )
            dinov2_cost = compute_true_latent_cost(
                dinov2_latents[1:], dinov2_latents[0],
            )
            del dinov2_latents
            positions = np.stack(positions)
            distances = np.linalg.norm(
                positions - np.asarray(case["goal_xy"]), axis=1,
            )
            rows = [
                {
                    "seed": seed,
                    "candidate": candidate,
                    "actual_x": float(positions[offset, 0]),
                    "actual_y": float(positions[offset, 1]),
                    "actual_distance": float(distances[offset]),
                    "ijepa_mse": float(ijepa_cost[offset]),
                    "dinov2_mse": float(dinov2_cost[offset]),
                }
                for offset, candidate in enumerate(candidate_ids)
            ]
            rows_by_seed[seed] = rows
            summaries[seed] = summarize_geometry(rows)
            print(
                f"seed {seed}: n={len(rows)}, "
                f"I-JEPA rho={summaries[seed]['ijepa_distance_spearman']:.3f}, "
                f"DINOv2 rho={summaries[seed]['dinov2_distance_spearman']:.3f}",
                flush=True,
            )
    finally:
        environment.close()
    all_rows = [row for rows in rows_by_seed.values() for row in rows]
    with (output_dir / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    plot_geometry(rows_by_seed, summaries, output_dir / "comparison.png")
    base._write_json_atomically({
        "source_diagnostic": str(args.input_dir),
        "ijepa_checkpoint": str(ijepa_checkpoint),
        "dinov2_checkpoint": str(args.dinov2_checkpoint),
        "dinov2_feature": "x_norm_patchtokens",
        "seeds": args.seeds,
        "candidates_per_seed": args.candidates_per_seed,
        "sampling_seed": args.sampling_seed,
        "sampling_note": "Candidate actions were sampled uniformly from the original pre-scoring CEM population. Starts were selected by the preceding diagnostic.",
        "device": str(device),
        "precision": precision,
        "cases": {str(seed): summaries[seed] for seed in args.seeds},
    }, output_dir / "summary.json")
    print(f"Saved encoder comparison to {output_dir}")


if __name__ == "__main__":
    main()

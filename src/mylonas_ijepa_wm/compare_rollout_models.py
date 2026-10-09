"""Compare two world models on identical autoregressive rollout windows."""

import argparse
import json
from pathlib import Path
import time

import torch

from src.mylonas_ijepa_wm.evaluate_rollouts import (
    RolloutEvaluationDataset,
    autoregressive_rollout,
    decode_positions,
    intersect_episode_ids,
    load_model_and_metadata,
    predict_position_trajectory,
)
from src.mylonas_ijepa_wm.position_probe import load_probe_checkpoint
from src.mylonas_ijepa_wm.wm import (
    LATENTS_PATH,
    TRAJECTORIES_PATH,
    compile_model,
    resolve_device,
    resolve_precision,
    validate_checkpoint_metadata,
)


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE_CHECKPOINT = (
    MODULE_DIR / "world_models" / "full_default_v1" / "best.pt"
)
DEFAULT_CANDIDATE_CHECKPOINT = (
    MODULE_DIR / "world_models" / "full_deafult_v1_k3" / "best.pt"
)
DEFAULT_POSITION_PROBE_CHECKPOINT = (
    MODULE_DIR / "position_probe_runs" / "spatial_softmax_mlp" / "best.pt"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare decoded-position rollouts from two world models on "
            "the exact same windows."
        ),
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=DEFAULT_BASELINE_CHECKPOINT,
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=DEFAULT_CANDIDATE_CHECKPOINT,
    )
    parser.add_argument(
        "--position-probe-checkpoint",
        type=Path,
        default=DEFAULT_POSITION_PROBE_CHECKPOINT,
    )
    parser.add_argument("--latent-dir", type=Path, default=LATENTS_PATH)
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=TRAJECTORIES_PATH,
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
    )
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "fp32", "bf16"],
        default="auto",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plot-output", type=Path, default=None)
    parser.add_argument(
        "--trajectory-plot-output",
        type=Path,
        default=None,
    )
    return parser.parse_args(argv)


def validate_arguments(args):
    for name in [
        "baseline_checkpoint",
        "candidate_checkpoint",
        "position_probe_checkpoint",
    ]:
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    if not args.latent_dir.is_dir():
        raise NotADirectoryError(
            f"Latent directory not found: {args.latent_dir}"
        )
    if not args.trajectory_dir.is_dir():
        raise NotADirectoryError(
            f"Trajectory directory not found: {args.trajectory_dir}"
        )
    if args.horizon < 1:
        raise ValueError("horizon must be at least 1")
    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")


def compare_position_rollouts(
    baseline_model,
    candidate_model,
    position_probe,
    data_loader,
    device,
    num_hist,
    position_mean,
    position_std,
    precision="fp32",
    log_every=0,
):
    baseline_model.eval()
    candidate_model.eval()
    position_probe.eval()

    baseline_error_batches = []
    candidate_error_batches = []
    oracle_error_batches = []
    episode_id_batches = []
    start_batches = []
    num_samples = 0
    started_at = time.perf_counter()
    non_blocking = device.type == "cuda"

    with torch.inference_mode():
        for step, batch in enumerate(data_loader, start=1):
            latent_sequence = batch["latent_sequence"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            )
            actions = batch["actions"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            )
            target_positions = batch["positions"][:, num_hist:].to(
                device=device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            )
            target_latents = latent_sequence[:, num_hist:]

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
            ):
                baseline_latents = autoregressive_rollout(
                    baseline_model,
                    latent_sequence[:, :num_hist],
                    actions,
                )
                candidate_latents = autoregressive_rollout(
                    candidate_model,
                    latent_sequence[:, :num_hist],
                    actions,
                )

            decoded_sources = {
                "baseline": baseline_latents,
                "candidate": candidate_latents,
                "oracle": target_latents,
            }
            decoded_positions = {
                name: decode_positions(
                    position_probe=position_probe,
                    latents=latents,
                    position_mean=position_mean,
                    position_std=position_std,
                )
                for name, latents in decoded_sources.items()
            }

            baseline_error_batches.append(
                torch.linalg.vector_norm(
                    decoded_positions["baseline"] - target_positions,
                    dim=-1,
                ).cpu()
            )
            candidate_error_batches.append(
                torch.linalg.vector_norm(
                    decoded_positions["candidate"] - target_positions,
                    dim=-1,
                ).cpu()
            )
            oracle_error_batches.append(
                torch.linalg.vector_norm(
                    decoded_positions["oracle"] - target_positions,
                    dim=-1,
                ).cpu()
            )
            episode_id_batches.append(
                torch.as_tensor(batch["episode_id"]).cpu()
            )
            start_batches.append(torch.as_tensor(batch["start"]).cpu())
            num_samples += latent_sequence.shape[0]

            if log_every > 0 and step % log_every == 0:
                elapsed = time.perf_counter() - started_at
                print(
                    f"  batch {step}/{len(data_loader)} "
                    f"samples/s={num_samples / elapsed:.2f}"
                )

    if not baseline_error_batches:
        raise ValueError("data loader is empty")

    return {
        "baseline_errors": torch.cat(baseline_error_batches, dim=0),
        "candidate_errors": torch.cat(candidate_error_batches, dim=0),
        "oracle_errors": torch.cat(oracle_error_batches, dim=0),
        "episode_ids": torch.cat(episode_id_batches, dim=0),
        "starts": torch.cat(start_batches, dim=0),
    }


def _summarize_error_pair(baseline_errors, candidate_errors):
    improvements = baseline_errors - candidate_errors
    return {
        "baseline_mean_error": baseline_errors.mean().item(),
        "baseline_median_error": torch.quantile(
            baseline_errors,
            0.5,
        ).item(),
        "candidate_mean_error": candidate_errors.mean().item(),
        "candidate_median_error": torch.quantile(
            candidate_errors,
            0.5,
        ).item(),
        "mean_improvement": improvements.mean().item(),
        "median_improvement": torch.quantile(improvements, 0.5).item(),
        "p10_improvement": torch.quantile(improvements, 0.10).item(),
        "p90_improvement": torch.quantile(improvements, 0.90).item(),
        "percent_improved": (
            improvements.gt(0).float().mean().mul(100).item()
        ),
        "percent_worse": (
            improvements.lt(0).float().mean().mul(100).item()
        ),
        "percent_tied": (
            improvements.eq(0).float().mean().mul(100).item()
        ),
    }


def summarize_improvements(baseline_errors, candidate_errors):
    baseline_errors = torch.as_tensor(baseline_errors).float()
    candidate_errors = torch.as_tensor(candidate_errors).float()

    if baseline_errors.shape != candidate_errors.shape:
        raise ValueError("baseline and candidate errors must have same shape")
    if baseline_errors.ndim != 2 or baseline_errors.shape[0] == 0:
        raise ValueError("errors must have shape [samples, horizons]")

    per_horizon = []
    for horizon_index in range(baseline_errors.shape[1]):
        row = _summarize_error_pair(
            baseline_errors[:, horizon_index],
            candidate_errors[:, horizon_index],
        )
        row["horizon"] = horizon_index + 1
        per_horizon.append(row)

    return {
        "num_samples": baseline_errors.shape[0],
        "improvement_definition": (
            "baseline position error - candidate position error; "
            "positive means the candidate is better"
        ),
        "final_horizon": _summarize_error_pair(
            baseline_errors[:, -1],
            candidate_errors[:, -1],
        ),
        "mean_over_horizons": _summarize_error_pair(
            baseline_errors.mean(dim=1),
            candidate_errors.mean(dim=1),
        ),
        "per_horizon": per_horizon,
    }


def select_representative_indices(improvements):
    improvements = torch.as_tensor(improvements).float().flatten()
    if improvements.numel() < 4:
        raise ValueError("at least four samples are required")

    selected = {
        "best": int(improvements.argmax()),
        "worst": int(improvements.argmin()),
    }
    used = set(selected.values())

    median = torch.quantile(improvements, 0.5)
    median_distances = (improvements - median).abs()
    for index in used:
        median_distances[index] = torch.inf
    selected["median"] = int(median_distances.argmin())
    used.add(selected["median"])

    zero_distances = improvements.abs()
    for index in used:
        zero_distances[index] = torch.inf
    selected["unchanged"] = int(zero_distances.argmin())

    return {
        name: selected[name]
        for name in ["best", "median", "unchanged", "worst"]
    }


def save_comparison_plot(summary, final_improvements, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary["per_horizon"]
    horizons = [row["horizon"] for row in rows]
    final_improvements = torch.as_tensor(final_improvements).numpy()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 4.5),
        constrained_layout=True,
    )

    axes[0].plot(
        horizons,
        [row["baseline_mean_error"] for row in rows],
        "o-",
        label="Baseline",
    )
    axes[0].plot(
        horizons,
        [row["candidate_mean_error"] for row in rows],
        "o-",
        label="Candidate",
    )
    axes[0].set_xlabel("Rollout horizon")
    axes[0].set_ylabel("Mean decoded-position error")
    axes[0].set_title("Aggregate position error")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        horizons,
        [row["mean_improvement"] for row in rows],
        "o-",
    )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("Rollout horizon")
    axes[1].set_ylabel("Baseline error - candidate error")
    axes[1].set_title("Mean improvement")
    axes[1].grid(alpha=0.25)

    axes[2].hist(final_improvements, bins=30, alpha=0.85)
    axes[2].axvline(0.0, color="black", linewidth=1)
    axes[2].set_xlabel("Baseline error - candidate error")
    axes[2].set_ylabel("Number of rollout windows")
    axes[2].set_title("Final-horizon improvement distribution")
    axes[2].grid(alpha=0.25)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_representative_trajectory_plot(
    trajectories,
    baseline_label,
    candidate_label,
    output_path,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not trajectories:
        raise ValueError("at least one trajectory is required")

    num_columns = min(2, len(trajectories))
    num_rows = (len(trajectories) + num_columns - 1) // num_columns
    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(6.5 * num_columns, 5.5 * num_rows),
        constrained_layout=True,
        squeeze=False,
    )

    styles = [
        ("true", "True", "o-", "black", 4),
        ("oracle", "Real latent decoded", "d:", "0.45", 2),
        (
            "baseline_autoregressive",
            f"{baseline_label} autoregressive",
            "x-",
            "tab:orange",
            4,
        ),
        (
            "candidate_autoregressive",
            f"{candidate_label} autoregressive",
            "+-",
            "tab:blue",
            4,
        ),
        (
            "baseline_teacher_forced",
            f"{baseline_label} teacher-forced",
            "^--",
            "peachpuff",
            1,
        ),
        (
            "candidate_teacher_forced",
            f"{candidate_label} teacher-forced",
            "v--",
            "lightskyblue",
            1,
        ),
    ]

    for axis, trajectory in zip(axes.flat, trajectories):
        for key, label, line_style, color, zorder in styles:
            positions = trajectory[key].numpy()
            axis.plot(
                positions[:, 0],
                positions[:, 1],
                line_style,
                color=color,
                label=label,
                zorder=zorder,
            )

        true_positions = trajectory["true"].numpy()
        axis.scatter(
            true_positions[0, 0],
            true_positions[0, 1],
            marker="s",
            s=65,
            color="black",
            label="Start",
            zorder=5,
        )
        axis.set_xlabel("Maze x")
        axis.set_ylabel("Maze y")
        axis.set_title(
            f"{trajectory['category'].replace('_', ' ').title()}: "
            f"episode {trajectory['episode_id']:04d}, "
            f"start {trajectory['start']}\n"
            f"mean improvement={trajectory['improvement']:+.4f}"
        )
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25)

    for axis in axes.flat[len(trajectories):]:
        axis.set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=min(4, len(labels)),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _build_representative_trajectory(
    category,
    improvement,
    sample,
    baseline_model,
    candidate_model,
    position_probe,
    device,
    num_hist,
    position_mean,
    position_std,
    precision,
):
    arguments = {
        "position_probe": position_probe,
        "sample": sample,
        "device": device,
        "num_hist": num_hist,
        "position_mean": position_mean,
        "position_std": position_std,
        "precision": precision,
    }
    baseline = predict_position_trajectory(
        model=baseline_model,
        **arguments,
    )
    candidate = predict_position_trajectory(
        model=candidate_model,
        **arguments,
    )
    return {
        "category": category,
        "improvement": float(improvement),
        "episode_id": candidate["episode_id"],
        "start": candidate["start"],
        "true": candidate["true"],
        "oracle": candidate["oracle"],
        "baseline_autoregressive": baseline["autoregressive"],
        "candidate_autoregressive": candidate["autoregressive"],
        "baseline_teacher_forced": baseline["teacher_forced"],
        "candidate_teacher_forced": candidate["teacher_forced"],
    }


def main(argv=None):
    args = parse_args(argv)
    validate_arguments(args)

    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    baseline_model, baseline_metadata, baseline_epoch = (
        load_model_and_metadata(args.baseline_checkpoint, device)
    )
    candidate_model, candidate_metadata, candidate_epoch = (
        load_model_and_metadata(args.candidate_checkpoint, device)
    )
    validate_checkpoint_metadata(
        saved_metadata=baseline_metadata,
        current_metadata=candidate_metadata,
        include_training_objective=False,
    )
    baseline_model = compile_model(
        baseline_model,
        enabled=args.compile,
    )
    candidate_model = compile_model(
        candidate_model,
        enabled=args.compile,
    )

    position_probe, probe_metadata, probe_epoch = load_probe_checkpoint(
        args.position_probe_checkpoint,
        device,
    )
    split_key = f"{args.split}_ids"
    world_model_episode_ids = baseline_metadata.get(split_key)
    probe_episode_ids = probe_metadata.get(split_key)
    if world_model_episode_ids is None:
        raise ValueError(
            f"world-model checkpoint does not contain {split_key}"
        )
    if probe_episode_ids is None:
        raise ValueError(
            f"position-probe checkpoint does not contain {split_key}"
        )
    episode_ids = intersect_episode_ids(
        world_model_episode_ids,
        probe_episode_ids,
    )

    model_configuration = baseline_metadata["model_configuration"]
    num_hist = model_configuration["num_hist"]
    action_mean = None
    action_std = None
    if baseline_metadata.get("normalize_actions", False):
        action_mean = baseline_metadata.get("action_mean")
        action_std = baseline_metadata.get("action_std")
        if action_mean is None or action_std is None:
            raise ValueError(
                "normalized-action checkpoint is missing action statistics"
            )

    dataset = RolloutEvaluationDataset(
        latent_dir=args.latent_dir,
        trajectory_dir=args.trajectory_dir,
        episode_ids=episode_ids,
        num_hist=num_hist,
        horizon=args.horizon,
        action_mean=action_mean,
        action_std=action_std,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    print(f"Baseline epoch: {baseline_epoch + 1}")
    print(f"Candidate epoch: {candidate_epoch + 1}")
    print(f"Position probe epoch: {probe_epoch + 1}")
    print(f"Split: {args.split} ({len(episode_ids)} episodes)")
    print(f"Rollout windows: {len(dataset)}")
    print(f"History: {num_hist}, horizon: {args.horizon}")
    print(f"Device: {device}, precision: {precision}")

    comparison = compare_position_rollouts(
        baseline_model=baseline_model,
        candidate_model=candidate_model,
        position_probe=position_probe,
        data_loader=data_loader,
        device=device,
        num_hist=num_hist,
        position_mean=probe_metadata["position_mean"],
        position_std=probe_metadata["position_std"],
        precision=precision,
        log_every=args.log_every,
    )
    summary = summarize_improvements(
        comparison["baseline_errors"],
        comparison["candidate_errors"],
    )

    mean_improvements = (
        comparison["baseline_errors"].mean(dim=1)
        - comparison["candidate_errors"].mean(dim=1)
    )
    selected_indices = select_representative_indices(mean_improvements)
    trajectories = []
    representative_cases = []
    for category, index in selected_indices.items():
        trajectory = _build_representative_trajectory(
            category=category,
            improvement=mean_improvements[index],
            sample=dataset[index],
            baseline_model=baseline_model,
            candidate_model=candidate_model,
            position_probe=position_probe,
            device=device,
            num_hist=num_hist,
            position_mean=probe_metadata["position_mean"],
            position_std=probe_metadata["position_std"],
            precision=precision,
        )
        trajectories.append(trajectory)
        representative_cases.append({
            "category": category,
            "dataset_index": index,
            "episode_id": trajectory["episode_id"],
            "start": trajectory["start"],
            "mean_improvement": mean_improvements[index].item(),
            "final_horizon_improvement": (
                comparison["baseline_errors"][index, -1]
                - comparison["candidate_errors"][index, -1]
            ).item(),
        })

    windows = []
    for index in range(len(dataset)):
        baseline_errors = comparison["baseline_errors"][index]
        candidate_errors = comparison["candidate_errors"][index]
        windows.append({
            "episode_id": int(comparison["episode_ids"][index]),
            "start": int(comparison["starts"][index]),
            "baseline_mean_error": baseline_errors.mean().item(),
            "candidate_mean_error": candidate_errors.mean().item(),
            "mean_improvement": (
                baseline_errors.mean() - candidate_errors.mean()
            ).item(),
            "baseline_final_error": baseline_errors[-1].item(),
            "candidate_final_error": candidate_errors[-1].item(),
            "final_horizon_improvement": (
                baseline_errors[-1] - candidate_errors[-1]
            ).item(),
        })

    metrics = {
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "baseline_epoch": baseline_epoch,
        "candidate_checkpoint": str(args.candidate_checkpoint),
        "candidate_epoch": candidate_epoch,
        "position_probe_checkpoint": str(
            args.position_probe_checkpoint
        ),
        "position_probe_epoch": probe_epoch,
        "split": args.split,
        "num_episodes": len(episode_ids),
        "num_hist": num_hist,
        "max_horizon": args.horizon,
        "selection_metric": "mean improvement over all horizons",
        "summary": summary,
        "representative_cases": representative_cases,
        "windows": windows,
    }

    output_path = args.output
    if output_path is None:
        output_path = args.candidate_checkpoint.parent / (
            f"comparison_{args.split}_h{args.horizon}.json"
        )
    plot_output = args.plot_output
    if plot_output is None:
        plot_output = output_path.with_suffix(".png")
    trajectory_plot_output = args.trajectory_plot_output
    if trajectory_plot_output is None:
        trajectory_plot_output = output_path.with_name(
            output_path.stem + "_trajectories.png"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2)
    save_comparison_plot(
        summary=summary,
        final_improvements=(
            comparison["baseline_errors"][:, -1]
            - comparison["candidate_errors"][:, -1]
        ),
        output_path=plot_output,
    )
    save_representative_trajectory_plot(
        trajectories=trajectories,
        baseline_label=args.baseline_checkpoint.parent.name,
        candidate_label=args.candidate_checkpoint.parent.name,
        output_path=trajectory_plot_output,
    )

    final_summary = summary["final_horizon"]
    mean_summary = summary["mean_over_horizons"]
    print("\nPositive improvement means the candidate is better.")
    print(
        "Final horizon: "
        f"{final_summary['percent_improved']:.1f}% improved, "
        f"median={final_summary['median_improvement']:+.6f}, "
        f"p10={final_summary['p10_improvement']:+.6f}, "
        f"p90={final_summary['p90_improvement']:+.6f}"
    )
    print(
        "Mean over horizons: "
        f"{mean_summary['percent_improved']:.1f}% improved, "
        f"median={mean_summary['median_improvement']:+.6f}, "
        f"p10={mean_summary['p10_improvement']:+.6f}, "
        f"p90={mean_summary['p90_improvement']:+.6f}"
    )
    print(f"Saved metrics to: {output_path}")
    print(f"Saved summary plot to: {plot_output}")
    print(
        "Saved representative trajectories to: "
        f"{trajectory_plot_output}"
    )


if __name__ == "__main__":
    main()

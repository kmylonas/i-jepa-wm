"""Measure visual-token rollout error for a top-down proprioceptive world model."""

import argparse
import json
from pathlib import Path
import time

import torch

from src.mylonas_ijepa_wm import wm, wm_proprio
from src.mylonas_ijepa_wm.audit_wall_dataset import classify_transition


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--latent-dir", type=Path,
        default=wm_proprio.DEFAULT_CACHE / "latents",
    )
    parser.add_argument(
        "--trajectory-dir", type=Path,
        default=wm_proprio.DEFAULT_CACHE / "trajectories",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--by-wall-layout", action="store_true",
                        help="Report separate rollout and collision-event metrics for each Wall layout.")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot-output", type=Path)
    return parser.parse_args(argv)


def group_episode_ids_by_wall_layout(
    episode_ids, trajectory_dir, expected_manifest_sha256=None, allowed_layouts=None,
):
    groups = {}
    allowed = None if allowed_layouts is None else {tuple(layout) for layout in allowed_layouts}
    for episode_id in episode_ids:
        trajectory = torch.load(
            Path(trajectory_dir) / f"episode_{episode_id:04d}.pt",
            map_location="cpu", weights_only=True,
        )
        layout = trajectory.get("wall_layout")
        if layout is None:
            raise ValueError(f"Episode {episode_id} has no Wall layout")
        if (expected_manifest_sha256 is not None
                and trajectory.get("manifest_sha256") != expected_manifest_sha256):
            raise ValueError(f"Episode {episode_id} manifest identity differs from checkpoint")
        if allowed is not None and tuple(layout) not in allowed:
            raise ValueError(f"Episode {episode_id} layout is outside checkpoint split")
        key = f"{layout[0]},{layout[1]}"
        groups.setdefault(key, []).append(episode_id)
    return groups


def transition_subset_indices(dataset, trajectory_dir):
    """Find rollout windows whose first predicted macro step contains the event."""
    subsets = {"solid_contact": [], "door_crossing": []}
    cache = {}
    for sample_index, (episode_id, start) in enumerate(dataset.samples):
        if episode_id not in cache:
            trajectory = torch.load(
                Path(trajectory_dir) / f"episode_{episode_id:04d}.pt",
                map_location="cpu", weights_only=True,
            )
            layout = trajectory.get("wall_layout")
            if layout is None:
                raise ValueError(f"Episode {episode_id} has no Wall layout")
            actions = trajectory["actions"].numpy()
            states = trajectory["states"].numpy()
            cache[episode_id] = [
                classify_transition(states[i], actions[i], states[i + 1], layout)
                for i in range(len(actions))
            ]
        labels = cache[episode_id]
        model_stride = dataset.action_group_sizes[episode_id]
        frame_step = dataset.model_frame_steps[episode_id]
        embedding_stride = model_stride // frame_step
        first_action = (start + (dataset.num_hist - 1) * frame_step) * embedding_stride
        last_action = first_action + model_stride
        window = labels[first_action:last_action]
        if "solid_contact" in window:
            subsets["solid_contact"].append(sample_index)
        if any(label.startswith("door_crossing_") for label in window):
            subsets["door_crossing"].append(sample_index)
    return {key: indices for key, indices in subsets.items() if indices}


def teacher_forced_rollout(model, latent_window, state_window, actions, num_hist):
    """Predict future visual tokens and states from real histories."""
    visual_predictions = []
    state_predictions = []
    for step in range(latent_window.shape[1] - num_hist):
        visual, state = model(
            latent_window[:, step:step + num_hist],
            state_window[:, step:step + num_hist],
            actions[:, step:step + num_hist],
        )
        visual_predictions.append(visual[:, -1])
        state_predictions.append(state[:, -1])
    return (
        torch.stack(visual_predictions, dim=1),
        torch.stack(state_predictions, dim=1),
    )


def evaluate_model(
    model, data_loader, device, num_hist,
    precision="fp32", log_every=0, proprio_loss_weight=1.0,
):
    if proprio_loss_weight <= 0:
        raise ValueError("proprio_loss_weight must be positive")
    model.eval()
    methods = ("autoregressive", "teacher_forced", "persistence")
    sums = {
        modality: {name: None for name in methods}
        for modality in ("latent", "proprio")
    }
    latent_elements_per_horizon = 0
    proprio_elements_per_horizon = 0
    num_samples = 0
    started_at = time.perf_counter()
    non_blocking = device.type == "cuda"

    with torch.inference_mode():
        for step, batch in enumerate(data_loader, start=1):
            latents = batch["latent_window"].to(
                device=device, dtype=torch.float32, non_blocking=non_blocking,
            )
            states = batch["state_window"].to(
                device=device, dtype=torch.float32, non_blocking=non_blocking,
            )
            actions = batch["actions"].to(
                device=device, dtype=torch.float32, non_blocking=non_blocking,
            )
            latent_targets = latents[:, num_hist:]
            state_targets = states[:, num_hist:]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
            ):
                autoregressive = wm_proprio.autoregressive_rollout(
                    model, latents[:, :num_hist], states[:, :num_hist], actions,
                )
                teacher_forced = teacher_forced_rollout(
                    model, latents, states, actions, num_hist,
                )
            persistence = (
                latents[:, num_hist - 1:num_hist].expand_as(latent_targets),
                states[:, num_hist - 1:num_hist].expand_as(state_targets),
            )

            for name, (visual_predictions, state_predictions) in (
                ("autoregressive", autoregressive),
                ("teacher_forced", teacher_forced),
                ("persistence", persistence),
            ):
                for modality, predictions, targets, dimensions in (
                    ("latent", visual_predictions, latent_targets, (0, 2, 3)),
                    ("proprio", state_predictions, state_targets, (0, 2)),
                ):
                    batch_sse = (predictions.float() - targets).square().sum(dim=dimensions)
                    if sums[modality][name] is None:
                        sums[modality][name] = torch.zeros_like(batch_sse)
                    sums[modality][name] += batch_sse
            latent_elements_per_horizon += latent_targets[:, 0].numel()
            proprio_elements_per_horizon += state_targets[:, 0].numel()
            num_samples += latent_targets.shape[0]
            if log_every > 0 and step % log_every == 0:
                elapsed = time.perf_counter() - started_at
                print(f"  batch {step}/{len(data_loader)} samples/s={num_samples / elapsed:.2f}")

    if num_samples == 0:
        raise ValueError("no rollout windows for this split and horizon")
    mse = {
        modality: {
            name: (value / elements_per_horizon).cpu().tolist()
            for name, value in sums[modality].items()
        }
        for modality, elements_per_horizon in (
            ("latent", latent_elements_per_horizon),
            ("proprio", proprio_elements_per_horizon),
        )
    }
    return {
        "num_samples": num_samples,
        "proprio_loss_weight": proprio_loss_weight,
        "per_horizon": [
            {
                "horizon": index + 1,
                **{
                    f"{name}_mse": mse["latent"][name][index]
                    for name in methods
                },
                **{
                    f"{name}_proprio_mse": mse["proprio"][name][index]
                    for name in methods
                },
                **{
                    f"{name}_combined_mse": (
                        mse["latent"][name][index]
                        + proprio_loss_weight * mse["proprio"][name][index]
                    )
                    for name in methods
                },
            }
            for index in range(len(mse["latent"]["autoregressive"]))
        ],
    }


def save_metrics_plot(metrics, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = metrics["per_horizon"]
    horizons = [row["horizon"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for axis, suffix, title, ylabel in (
        (axes[0], "mse", "Visual latents", "Visual latent MSE"),
        (axes[1], "proprio_mse", "Proprioception", "Normalized state MSE"),
        (
            axes[2], "combined_mse", "Combined",
            f"Latent + {metrics['proprio_loss_weight']:g} × state MSE",
        ),
    ):
        for name, label in (
            ("autoregressive", "Autoregressive"),
            ("teacher_forced", "Teacher-forced"),
            ("persistence", "Persistence"),
        ):
            axis.plot(
                horizons, [row[f"{name}_{suffix}"] for row in rows],
                marker="o", label=label,
            )
        axis.set_xticks(horizons)
        axis.set_xlabel("Rollout horizon (macro steps)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend()
        axis.grid(alpha=0.25)
    figure.suptitle(f"Proprioceptive world model — {metrics['split']} split")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    for directory in (args.latent_dir, args.trajectory_dir):
        if not directory.is_dir():
            raise NotADirectoryError(directory)
    if args.horizon < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("horizon and batch-size must be positive; num-workers cannot be negative")

    device = wm.resolve_device(args.device)
    precision = wm.resolve_precision(args.precision, device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model, metadata, epoch = wm_proprio.load_model_and_metadata(args.checkpoint, device)
    model = wm.compile_model(model, enabled=args.compile)
    proprio_loss_weight = metadata.get("proprio_loss_weight")
    if proprio_loss_weight is None or proprio_loss_weight <= 0:
        raise ValueError("checkpoint is missing a positive proprio_loss_weight")
    episode_ids = metadata.get(f"{args.split}_ids")
    if not episode_ids:
        raise ValueError(f"checkpoint has no {args.split}_ids")
    if metadata.get("normalize_actions", False):
        action_mean = metadata.get("action_mean")
        action_std = metadata.get("action_std")
        if action_mean is None or action_std is None:
            raise ValueError("normalized-action checkpoint is missing action statistics")
    else:
        action_mean = action_std = None

    num_hist = metadata["model_configuration"]["num_hist"]
    dataset = wm_proprio.ProprioWorldModelDataset(
        latent_dir=args.latent_dir,
        trajectory_dir=args.trajectory_dir,
        episode_ids=episode_ids,
        num_hist=num_hist,
        rollout_steps=args.horizon,
        action_mean=action_mean,
        action_std=action_std,
        state_mean=metadata["state_mean"],
        state_std=metadata["state_std"],
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    print(f"Checkpoint epoch: {epoch + 1}")
    print(f"Split: {args.split} ({len(episode_ids)} episodes)")
    print(f"Rollout samples: {len(dataset)}")
    print(f"History: {num_hist}, horizon: {args.horizon}")
    print(f"Device: {device}, precision: {precision}")
    metrics = evaluate_model(
        model, loader, device, num_hist, precision=precision, log_every=args.log_every,
        proprio_loss_weight=proprio_loss_weight,
    )
    metrics.update({
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": epoch,
        "split": args.split,
        "num_episodes": len(episode_ids),
        "episode_ids": episode_ids,
        "num_hist": num_hist,
        "max_horizon": args.horizon,
    })
    if args.by_wall_layout:
        if metadata.get("environment") != "wall":
            raise ValueError("--by-wall-layout requires a Wall checkpoint")
        metrics["by_wall_layout"] = {}
        groups = group_episode_ids_by_wall_layout(
            episode_ids, args.trajectory_dir,
            expected_manifest_sha256=metadata.get("experiment_manifest_sha256"),
            allowed_layouts=(metadata.get("layout_splits") or {}).get(args.split),
        )
        for layout, group_ids in groups.items():
            group_dataset = wm_proprio.ProprioWorldModelDataset(
                latent_dir=args.latent_dir,
                trajectory_dir=args.trajectory_dir,
                episode_ids=group_ids,
                num_hist=num_hist,
                rollout_steps=args.horizon,
                action_mean=action_mean,
                action_std=action_std,
                state_mean=metadata["state_mean"],
                state_std=metadata["state_std"],
            )
            group_loader = torch.utils.data.DataLoader(
                group_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=device.type == "cuda",
                persistent_workers=args.num_workers > 0,
            )
            group_metrics = evaluate_model(
                model, group_loader, device, num_hist, precision=precision,
                proprio_loss_weight=proprio_loss_weight,
            )
            group_metrics["num_episodes"] = len(group_ids)
            group_metrics["episode_ids"] = group_ids
            group_metrics["transition_subsets"] = {}
            group_metrics["transition_subset_definition"] = (
                "The first predicted macro step contains a real contact or doorway crossing; "
                "later horizons measure rollout after that event."
            )
            for category, indices in transition_subset_indices(
                group_dataset, args.trajectory_dir,
            ).items():
                subset_loader = torch.utils.data.DataLoader(
                    torch.utils.data.Subset(group_dataset, indices),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=device.type == "cuda",
                    persistent_workers=args.num_workers > 0,
                )
                group_metrics["transition_subsets"][category] = evaluate_model(
                    model, subset_loader, device, num_hist, precision=precision,
                    proprio_loss_weight=proprio_loss_weight,
                )
            metrics["by_wall_layout"][layout] = group_metrics
    output_path = args.output or args.checkpoint.parent / f"rollout_{args.split}_h{args.horizon}_metrics.json"
    plot_path = args.plot_output or output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_metrics_plot(metrics, plot_path)
    for label, suffix in (
        ("visual latent MSE", "mse"),
        ("normalized state MSE", "proprio_mse"),
        (f"combined MSE (state weight {proprio_loss_weight:g})", "combined_mse"),
    ):
        print(f"\n{label}")
        print("horizon  autoregressive  teacher-forced  persistence")
        for row in metrics["per_horizon"]:
            print(
                f"{row['horizon']:>7}  {row[f'autoregressive_{suffix}']:>14.6f}  "
                f"{row[f'teacher_forced_{suffix}']:>14.6f}  "
                f"{row[f'persistence_{suffix}']:>11.6f}"
            )
    print(f"Saved metrics to: {output_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()

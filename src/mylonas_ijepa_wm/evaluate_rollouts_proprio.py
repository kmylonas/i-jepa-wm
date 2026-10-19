"""Measure visual-token rollout error for a top-down proprioceptive world model."""

import argparse
import json
from pathlib import Path
import time

import torch

from src.mylonas_ijepa_wm import wm, wm_proprio


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


def teacher_forced_rollout(model, latent_window, state_window, actions, num_hist):
    """Predict each future latent from the corresponding real visual/state history."""
    predictions = []
    for step in range(latent_window.shape[1] - num_hist):
        visual, _ = model(
            latent_window[:, step:step + num_hist],
            state_window[:, step:step + num_hist],
            actions[:, step:step + num_hist],
        )
        predictions.append(visual[:, -1])
    return torch.stack(predictions, dim=1)


def evaluate_model(model, data_loader, device, num_hist, precision="fp32", log_every=0):
    model.eval()
    sums = {name: None for name in ("autoregressive", "teacher_forced", "persistence")}
    elements_per_horizon = 0
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
            targets = latents[:, num_hist:]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
            ):
                autoregressive, _ = wm_proprio.autoregressive_rollout(
                    model, latents[:, :num_hist], states[:, :num_hist], actions,
                )
                teacher_forced = teacher_forced_rollout(
                    model, latents, states, actions, num_hist,
                )
            persistence = latents[:, num_hist - 1:num_hist].expand_as(targets)

            for name, predictions in (
                ("autoregressive", autoregressive),
                ("teacher_forced", teacher_forced),
                ("persistence", persistence),
            ):
                batch_sse = (predictions.float() - targets).square().sum(dim=(0, 2, 3))
                if sums[name] is None:
                    sums[name] = torch.zeros_like(batch_sse)
                sums[name] += batch_sse
            elements_per_horizon += targets[:, 0].numel()
            num_samples += targets.shape[0]
            if log_every > 0 and step % log_every == 0:
                elapsed = time.perf_counter() - started_at
                print(f"  batch {step}/{len(data_loader)} samples/s={num_samples / elapsed:.2f}")

    if num_samples == 0:
        raise ValueError("no rollout windows for this split and horizon")
    mse = {
        name: (value / elements_per_horizon).cpu().tolist()
        for name, value in sums.items()
    }
    return {
        "num_samples": num_samples,
        "per_horizon": [
            {
                "horizon": index + 1,
                "autoregressive_mse": mse["autoregressive"][index],
                "teacher_forced_mse": mse["teacher_forced"][index],
                "persistence_mse": mse["persistence"][index],
            }
            for index in range(len(mse["autoregressive"]))
        ],
    }


def save_metrics_plot(metrics, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = metrics["per_horizon"]
    horizons = [row["horizon"] for row in rows]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for key, label in (
        ("autoregressive_mse", "Autoregressive"),
        ("teacher_forced_mse", "Teacher-forced"),
        ("persistence_mse", "Persistence"),
    ):
        axis.plot(horizons, [row[key] for row in rows], marker="o", label=label)
    axis.set_xticks(horizons)
    axis.set_xlabel("Rollout horizon (macro steps)")
    axis.set_ylabel("Visual latent MSE")
    axis.set_title(f"Proprioceptive world model — {metrics['split']} split")
    axis.legend()
    axis.grid(alpha=0.25)
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
    output_path = args.output or args.checkpoint.parent / f"rollout_{args.split}_h{args.horizon}_latent.json"
    plot_path = args.plot_output or output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_metrics_plot(metrics, plot_path)
    print("\nhorizon  autoregressive  teacher-forced  persistence")
    for row in metrics["per_horizon"]:
        print(
            f"{row['horizon']:>7}  {row['autoregressive_mse']:>14.6f}  "
            f"{row['teacher_forced_mse']:>14.6f}  {row['persistence_mse']:>11.6f}"
        )
    print(f"Saved metrics to: {output_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()

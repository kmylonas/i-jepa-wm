"""Visualize PointMaze position predictions from a trained probe."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.mylonas_ijepa_wm.position_probe import (
    LATENTS_PATH,
    REPO_ROOT,
    TRAJECTORIES_PATH,
    load_probe_checkpoint,
)
from src.mylonas_ijepa_wm.wm import resolve_device


DEFAULT_CHECKPOINT_PATH = (
    REPO_ROOT / "position_probe_runs" / "spatial_softmax" / "best.pt"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualize position-probe predictions.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )
    parser.add_argument("--latent-dir", type=Path, default=LATENTS_PATH)
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=TRAJECTORIES_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
    )
    parser.add_argument("--episode-id", type=int, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--frame-index", type=int, default=10)
    parser.add_argument("--max-points", type=int, default=1000)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    return parser.parse_args(argv)


def load_probe(checkpoint_path, device):
    model, metadata, _ = load_probe_checkpoint(
        checkpoint_path,
        device,
    )
    return model, metadata


def predict_episode(
    model,
    episode_id,
    latent_dir,
    trajectory_dir,
    position_mean,
    position_std,
    device,
):
    latents = torch.from_numpy(
        np.load(
            Path(latent_dir) / f"episode_{episode_id:04d}.npy",
            allow_pickle=False,
        )
    ).to(device=device, dtype=torch.float32)
    trajectory = torch.load(
        Path(trajectory_dir) / f"episode_{episode_id:04d}.pt",
        map_location="cpu",
        weights_only=True,
    )
    targets = trajectory["sampled_states"][:, :2].float()

    with torch.inference_mode():
        normalized_predictions = model(latents)
        probabilities = model.patch_probabilities(latents)
    predictions = (
        normalized_predictions.cpu() * position_std
        + position_mean
    )
    return predictions, targets, probabilities.cpu(), trajectory


def collect_split_predictions(
    model,
    episode_ids,
    latent_dir,
    trajectory_dir,
    position_mean,
    position_std,
    device,
):
    predictions = []
    targets = []
    sample_episode_ids = []

    for episode_id in episode_ids:
        episode_predictions, episode_targets, _, _ = predict_episode(
            model=model,
            episode_id=episode_id,
            latent_dir=latent_dir,
            trajectory_dir=trajectory_dir,
            position_mean=position_mean,
            position_std=position_std,
            device=device,
        )
        predictions.append(episode_predictions)
        targets.append(episode_targets)
        sample_episode_ids.append(
            torch.full((len(episode_targets),), episode_id)
        )

    return {
        "predictions": torch.cat(predictions),
        "targets": torch.cat(targets),
        "episode_ids": torch.cat(sample_episode_ids),
    }


def save_position_diagnostics(
    predictions,
    targets,
    episode_predictions,
    episode_targets,
    episode_id,
    output_path,
    max_points,
):
    predictions = predictions.numpy()
    targets = targets.numpy()
    episode_predictions = episode_predictions.numpy()
    episode_targets = episode_targets.numpy()

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
        constrained_layout=True,
    )

    for coordinate_index, coordinate_name in enumerate(["x", "y"]):
        axis = axes[0, coordinate_index]
        actual = targets[:, coordinate_index]
        predicted = predictions[:, coordinate_index]
        lower = min(actual.min(), predicted.min())
        upper = max(actual.max(), predicted.max())
        axis.scatter(actual, predicted, s=8, alpha=0.25)
        axis.plot([lower, upper], [lower, upper], "--", color="black")
        axis.set_xlabel(f"True {coordinate_name}")
        axis.set_ylabel(f"Predicted {coordinate_name}")
        axis.set_title(f"{coordinate_name}-coordinate parity")
        axis.set_aspect("equal", adjustable="box")

    axis = axes[1, 0]
    sample_count = min(max_points, len(targets))
    sample_indices = np.linspace(
        0,
        len(targets) - 1,
        sample_count,
        dtype=int,
    )
    for index in sample_indices:
        axis.plot(
            [targets[index, 0], predictions[index, 0]],
            [targets[index, 1], predictions[index, 1]],
            color="0.75",
            linewidth=0.4,
            alpha=0.4,
        )
    axis.scatter(
        targets[sample_indices, 0],
        targets[sample_indices, 1],
        s=8,
        label="True",
    )
    axis.scatter(
        predictions[sample_indices, 0],
        predictions[sample_indices, 1],
        s=10,
        marker="x",
        label="Predicted",
    )
    axis.set_xlabel("Maze x")
    axis.set_ylabel("Maze y")
    axis.set_title("Position errors")
    axis.set_aspect("equal", adjustable="box")
    axis.legend()

    axis = axes[1, 1]
    axis.plot(
        episode_targets[:, 0],
        episode_targets[:, 1],
        "o-",
        label="True",
    )
    axis.plot(
        episode_predictions[:, 0],
        episode_predictions[:, 1],
        "x-",
        label="Predicted",
    )
    for target, prediction in zip(
        episode_targets,
        episode_predictions,
    ):
        axis.plot(
            [target[0], prediction[0]],
            [target[1], prediction[1]],
            color="0.75",
            linewidth=0.7,
        )
    axis.set_xlabel("Maze x")
    axis.set_ylabel("Maze y")
    axis.set_title(f"Episode {episode_id:04d}")
    axis.set_aspect("equal", adjustable="box")
    axis.legend()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_attention_overlay(
    probabilities,
    prediction,
    target,
    raw_episode_path,
    raw_frame_index,
    grid_size,
    output_path,
):
    raw_episode = torch.load(
        raw_episode_path,
        map_location="cpu",
        weights_only=True,
    )
    frame = raw_episode["frames"][raw_frame_index].numpy()
    height, width = frame.shape[:2]
    heatmap = probabilities.reshape(grid_size, grid_size).numpy()

    figure, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
    axis.imshow(frame)
    axis.imshow(
        heatmap,
        extent=(0, width, height, 0),
        interpolation="bilinear",
        cmap="magma",
        alpha=0.55,
    )
    error = torch.linalg.vector_norm(prediction - target).item()
    axis.set_title(
        f"True=({target[0]:.2f}, {target[1]:.2f})  "
        f"Predicted=({prediction[0]:.2f}, {prediction[1]:.2f})  "
        f"Error={error:.2f}"
    )
    axis.axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.latent_dir.is_dir():
        raise NotADirectoryError(
            f"Latent directory not found: {args.latent_dir}"
        )
    if not args.trajectory_dir.is_dir():
        raise NotADirectoryError(
            f"Trajectory directory not found: {args.trajectory_dir}"
        )
    if args.max_points < 1:
        raise ValueError("max-points must be at least 1")

    output_dir = args.output_dir or args.checkpoint.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model, metadata = load_probe(args.checkpoint, device)
    position_mean = metadata["position_mean"].float()
    position_std = metadata["position_std"].float()
    split_ids = metadata[f"{args.split}_ids"]
    episode_id = args.episode_id
    if episode_id is None:
        episode_id = split_ids[0]

    split_predictions = collect_split_predictions(
        model=model,
        episode_ids=split_ids,
        latent_dir=args.latent_dir,
        trajectory_dir=args.trajectory_dir,
        position_mean=position_mean,
        position_std=position_std,
        device=device,
    )
    (
        episode_predictions,
        episode_targets,
        episode_probabilities,
        trajectory,
    ) = predict_episode(
        model=model,
        episode_id=episode_id,
        latent_dir=args.latent_dir,
        trajectory_dir=args.trajectory_dir,
        position_mean=position_mean,
        position_std=position_std,
        device=device,
    )

    diagnostics_path = output_dir / "position_diagnostics.png"
    save_position_diagnostics(
        predictions=split_predictions["predictions"],
        targets=split_predictions["targets"],
        episode_predictions=episode_predictions,
        episode_targets=episode_targets,
        episode_id=episode_id,
        output_path=diagnostics_path,
        max_points=args.max_points,
    )
    print(f"Saved position diagnostics to: {diagnostics_path}")

    if args.raw_dir is not None:
        raw_episode_path = (
            args.raw_dir / f"episode_{episode_id:04d}.pt"
        )
        if not raw_episode_path.is_file():
            raise FileNotFoundError(
                f"Raw episode not found: {raw_episode_path}"
            )
        if not 0 <= args.frame_index < len(episode_predictions):
            raise ValueError(
                f"frame-index must be between 0 and "
                f"{len(episode_predictions) - 1}"
            )

        frame_indices = trajectory.get("frame_indices")
        if frame_indices is None:
            raw_frame_index = args.frame_index * 5
        else:
            raw_frame_index = int(frame_indices[args.frame_index])
        overlay_path = output_dir / (
            f"attention_episode_{episode_id:04d}_"
            f"frame_{args.frame_index:02d}.png"
        )
        save_attention_overlay(
            probabilities=episode_probabilities[args.frame_index],
            prediction=episode_predictions[args.frame_index],
            target=episode_targets[args.frame_index],
            raw_episode_path=raw_episode_path,
            raw_frame_index=raw_frame_index,
            grid_size=model.grid_size,
            output_path=overlay_path,
        )
        print(f"Saved attention overlay to: {overlay_path}")


if __name__ == "__main__":
    main()

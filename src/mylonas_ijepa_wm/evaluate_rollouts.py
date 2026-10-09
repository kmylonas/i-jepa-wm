"""Evaluate autoregressive latent rollouts from a trained world model."""

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.mylonas_ijepa_wm.position_probe import (
    compute_position_metrics,
    load_probe_checkpoint,
)
from src.mylonas_ijepa_wm.vit import ViT
from src.mylonas_ijepa_wm.wm import (
    LATENTS_PATH,
    REPO_ROOT,
    TRAJECTORIES_PATH,
    compile_model,
    resolve_device,
    resolve_precision,
)


DEFAULT_CHECKPOINT_PATH = (
    REPO_ROOT / "world_model_runs" / "baseline" / "best.pt"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate multi-step world-model latent rollouts.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--position-probe-checkpoint",
        type=Path,
        default=None,
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
        default="val",
        help="Use validation while choosing rollout and planning settings.",
    )
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
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
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plot-output", type=Path, default=None)
    parser.add_argument("--num-trajectory-plots", type=int, default=4)
    parser.add_argument(
        "--trajectory-plot-output",
        type=Path,
        default=None,
    )
    return parser.parse_args(argv)


class RolloutEvaluationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        latent_dir,
        trajectory_dir,
        episode_ids,
        num_hist,
        horizon,
        action_mean=None,
        action_std=None,
        episode_cache_size=8,
    ):
        self.latent_dir = Path(latent_dir)
        self.trajectory_dir = Path(trajectory_dir)
        self.num_hist = num_hist
        self.horizon = horizon
        self.action_mean = action_mean
        self.action_std = action_std
        self.episode_cache_size = episode_cache_size

        if num_hist < 1:
            raise ValueError("num_hist must be at least 1")
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        if episode_cache_size < 1:
            raise ValueError("episode_cache_size must be at least 1")
        if (action_mean is None) != (action_std is None):
            raise ValueError(
                "action_mean and action_std must either both be set or both be None"
            )

        self.samples = []
        self.actions = {}
        self.positions = {}
        self._latent_cache = OrderedDict()
        sequence_length = num_hist + horizon

        for episode_id in episode_ids:
            latent_path = self._latent_path(episode_id)
            latents = np.load(
                latent_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            num_frames = len(latents)

            trajectory = torch.load(
                self._trajectory_path(episode_id),
                map_location="cpu",
                weights_only=True,
            )
            actions = trajectory["macro_actions"].float()
            positions = trajectory["sampled_states"][:, :2].float()

            if len(actions) < num_frames - 1:
                raise ValueError(
                    f"Episode {episode_id} has {num_frames} latent frames "
                    f"but only {len(actions)} macro actions"
                )
            if len(positions) != num_frames:
                raise ValueError(
                    f"Episode {episode_id} has {num_frames} latent frames "
                    f"but {len(positions)} sampled states"
                )

            self.actions[episode_id] = actions
            self.positions[episode_id] = positions
            for start in range(num_frames - sequence_length + 1):
                self.samples.append((episode_id, start))

    def _latent_path(self, episode_id):
        return self.latent_dir / f"episode_{episode_id:04d}.npy"

    def _trajectory_path(self, episode_id):
        return self.trajectory_dir / f"episode_{episode_id:04d}.pt"

    def __len__(self):
        return len(self.samples)

    def load_latents(self, episode_id):
        if episode_id in self._latent_cache:
            self._latent_cache.move_to_end(episode_id)
            return self._latent_cache[episode_id]

        latents = np.load(
            self._latent_path(episode_id),
            mmap_mode="r",
            allow_pickle=False,
        )
        self._latent_cache[episode_id] = latents

        if len(self._latent_cache) > self.episode_cache_size:
            self._latent_cache.popitem(last=False)

        return latents

    def __getitem__(self, index):
        episode_id, start = self.samples[index]
        sequence_end = start + self.num_hist + self.horizon
        action_end = sequence_end - 1

        latent_sequence = np.array(
            self.load_latents(episode_id)[start:sequence_end],
            copy=True,
        )
        actions = self.actions[episode_id][start:action_end]

        if self.action_mean is not None:
            actions = (
                actions - self.action_mean
            ) / self.action_std

        return {
            "latent_sequence": torch.from_numpy(latent_sequence),
            "actions": actions,
            "positions": self.positions[episode_id][
                start:sequence_end
            ],
            "episode_id": episode_id,
            "start": start,
        }


def intersect_episode_ids(world_model_ids, probe_ids):
    probe_id_set = set(probe_ids)
    shared_ids = [
        episode_id
        for episode_id in world_model_ids
        if episode_id in probe_id_set
    ]
    if not shared_ids:
        raise ValueError(
            "world model and position probe have no shared episodes"
        )
    return shared_ids


def autoregressive_rollout(model, latent_history, actions):
    num_hist = latent_history.shape[1]
    horizon = actions.shape[1] - num_hist + 1

    if horizon < 1:
        raise ValueError(
            "actions must contain num_hist + horizon - 1 steps"
        )

    predictions = []
    history = latent_history

    for step in range(horizon):
        action_window = actions[:, step:step + num_hist]
        predicted_window = model(history, action_window)
        next_latent = predicted_window[:, -1]
        predictions.append(next_latent)
        history = torch.cat(
            [history[:, 1:], next_latent.unsqueeze(1)],
            dim=1,
        )

    return torch.stack(predictions, dim=1)


def teacher_forced_rollout(
    model,
    latent_sequence,
    actions,
    num_hist,
):
    horizon = latent_sequence.shape[1] - num_hist
    predictions = []

    for step in range(horizon):
        latent_history = latent_sequence[:, step:step + num_hist]
        action_window = actions[:, step:step + num_hist]
        predicted_window = model(latent_history, action_window)
        predictions.append(predicted_window[:, -1])

    return torch.stack(predictions, dim=1)


def decode_positions(
    position_probe,
    latents,
    position_mean,
    position_std,
):
    leading_shape = latents.shape[:-2]
    flattened_latents = latents.reshape(
        -1,
        latents.shape[-2],
        latents.shape[-1],
    ).float()
    normalized_positions = position_probe(flattened_latents)
    position_mean = position_mean.to(
        device=normalized_positions.device,
        dtype=normalized_positions.dtype,
    )
    position_std = position_std.to(
        device=normalized_positions.device,
        dtype=normalized_positions.dtype,
    )
    positions = normalized_positions * position_std + position_mean
    return positions.reshape(*leading_shape, 2)


def predict_position_trajectory(
    model,
    position_probe,
    sample,
    device,
    num_hist,
    position_mean,
    position_std,
    precision="fp32",
):
    latent_sequence = sample["latent_sequence"].unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )
    actions = sample["actions"].unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )
    positions = sample["positions"].float()

    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16",
        ):
            autoregressive_latents = autoregressive_rollout(
                model,
                latent_sequence[:, :num_hist],
                actions,
            )
            teacher_forced_latents = teacher_forced_rollout(
                model,
                latent_sequence,
                actions,
                num_hist,
            )

        autoregressive_positions = decode_positions(
            position_probe=position_probe,
            latents=autoregressive_latents,
            position_mean=position_mean,
            position_std=position_std,
        )[0].cpu()
        teacher_forced_positions = decode_positions(
            position_probe=position_probe,
            latents=teacher_forced_latents,
            position_mean=position_mean,
            position_std=position_std,
        )[0].cpu()
        oracle_positions = decode_positions(
            position_probe=position_probe,
            latents=latent_sequence[:, num_hist:],
            position_mean=position_mean,
            position_std=position_std,
        )[0].cpu()

    start_position = positions[num_hist - 1:num_hist]
    return {
        "episode_id": int(sample["episode_id"]),
        "start": int(sample["start"]),
        "true": positions[num_hist - 1:].cpu(),
        "oracle": torch.cat(
            [start_position, oracle_positions],
            dim=0,
        ),
        "autoregressive": torch.cat(
            [start_position, autoregressive_positions],
            dim=0,
        ),
        "teacher_forced": torch.cat(
            [start_position, teacher_forced_positions],
            dim=0,
        ),
    }


def evaluate_model(
    model,
    data_loader,
    device,
    num_hist,
    precision="fp32",
    log_every=0,
    position_probe=None,
    position_mean=None,
    position_std=None,
):
    if position_probe is not None and (
        position_mean is None or position_std is None
    ):
        raise ValueError(
            "position mean and std are required with a position probe"
        )

    model.eval()
    if position_probe is not None:
        position_probe.eval()
    autoregressive_sse = None
    teacher_forced_sse = None
    persistence_sse = None
    elements_per_horizon = 0
    num_samples = 0
    decoded_position_batches = None
    target_position_batches = []
    if position_probe is not None:
        decoded_position_batches = {
            "oracle": [],
            "autoregressive": [],
            "teacher_forced": [],
            "persistence": [],
        }
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
            targets = latent_sequence[:, num_hist:]

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
            ):
                autoregressive_predictions = autoregressive_rollout(
                    model,
                    latent_sequence[:, :num_hist],
                    actions,
                )
                teacher_forced_predictions = teacher_forced_rollout(
                    model,
                    latent_sequence,
                    actions,
                    num_hist,
                )

            persistence_predictions = latent_sequence[
                :, num_hist - 1:num_hist
            ].expand_as(targets)

            if position_probe is not None:
                target_positions = batch["positions"][:, num_hist:].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=non_blocking,
                )
                latent_sources = {
                    "oracle": targets,
                    "autoregressive": autoregressive_predictions,
                    "teacher_forced": teacher_forced_predictions,
                    "persistence": persistence_predictions,
                }
                for name, source_latents in latent_sources.items():
                    decoded = decode_positions(
                        position_probe=position_probe,
                        latents=source_latents,
                        position_mean=position_mean,
                        position_std=position_std,
                    )
                    decoded_position_batches[name].append(
                        decoded.cpu()
                    )
                target_position_batches.append(target_positions.cpu())

            autoregressive_error = (
                autoregressive_predictions.float() - targets
            ).square()
            teacher_forced_error = (
                teacher_forced_predictions.float() - targets
            ).square()
            persistence_error = (
                persistence_predictions - targets
            ).square()

            reduce_dimensions = tuple(
                dimension
                for dimension in range(autoregressive_error.ndim)
                if dimension != 1
            )
            batch_autoregressive_sse = autoregressive_error.sum(
                dim=reduce_dimensions
            )
            batch_teacher_forced_sse = teacher_forced_error.sum(
                dim=reduce_dimensions
            )
            batch_persistence_sse = persistence_error.sum(
                dim=reduce_dimensions
            )

            if autoregressive_sse is None:
                autoregressive_sse = torch.zeros_like(
                    batch_autoregressive_sse
                )
                teacher_forced_sse = torch.zeros_like(
                    batch_teacher_forced_sse
                )
                persistence_sse = torch.zeros_like(
                    batch_persistence_sse
                )

            autoregressive_sse += batch_autoregressive_sse
            teacher_forced_sse += batch_teacher_forced_sse
            persistence_sse += batch_persistence_sse
            elements_per_horizon += targets[:, 0].numel()
            num_samples += targets.shape[0]

            if log_every > 0 and step % log_every == 0:
                elapsed = time.perf_counter() - started_at
                print(
                    f"  batch {step}/{len(data_loader)} "
                    f"samples/s={num_samples / elapsed:.2f}"
                )

    if num_samples == 0:
        raise ValueError("data loader is empty")

    autoregressive_mse = (
        autoregressive_sse / elements_per_horizon
    ).cpu().tolist()
    teacher_forced_mse = (
        teacher_forced_sse / elements_per_horizon
    ).cpu().tolist()
    persistence_mse = (
        persistence_sse / elements_per_horizon
    ).cpu().tolist()

    metrics = {
        "num_samples": num_samples,
        "per_horizon": [
            {
                "horizon": horizon + 1,
                "autoregressive_mse": autoregressive_mse[horizon],
                "teacher_forced_mse": teacher_forced_mse[horizon],
                "persistence_mse": persistence_mse[horizon],
            }
            for horizon in range(len(autoregressive_mse))
        ],
    }

    if position_probe is not None:
        target_positions = torch.cat(target_position_batches, dim=0)
        decoded_positions = {
            name: torch.cat(batches, dim=0)
            for name, batches in decoded_position_batches.items()
        }
        for horizon_index, row in enumerate(metrics["per_horizon"]):
            row["position"] = {}
            for name, predictions in decoded_positions.items():
                row["position"][name] = compute_position_metrics(
                    predictions=predictions[:, horizon_index],
                    targets=target_positions[:, horizon_index],
                    baseline_position=position_mean.cpu(),
                )

    return metrics


def load_model_and_metadata(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    metadata = checkpoint.get("metadata", {})
    model_configuration = metadata.get("model_configuration")
    if model_configuration is None:
        raise ValueError(
            "checkpoint metadata does not contain model_configuration"
        )

    model = ViT(**model_configuration)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, metadata, checkpoint["epoch"]


def validate_arguments(args):
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )
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
    if args.num_trajectory_plots < 1:
        raise ValueError("num-trajectory-plots must be at least 1")
    if (
        args.position_probe_checkpoint is not None
        and not args.position_probe_checkpoint.is_file()
    ):
        raise FileNotFoundError(
            "Position probe checkpoint not found: "
            f"{args.position_probe_checkpoint}"
        )
    if (
        args.plot_output is not None
        and args.position_probe_checkpoint is None
    ):
        raise ValueError(
            "plot-output requires position-probe-checkpoint"
        )
    if (
        args.trajectory_plot_output is not None
        and args.position_probe_checkpoint is None
    ):
        raise ValueError(
            "trajectory-plot-output requires "
            "position-probe-checkpoint"
        )


def print_metrics(metrics):
    print("\nhorizon  autoregressive  teacher-forced  persistence")
    for row in metrics["per_horizon"]:
        print(
            f"{row['horizon']:>7}  "
            f"{row['autoregressive_mse']:>14.6f}  "
            f"{row['teacher_forced_mse']:>14.6f}  "
            f"{row['persistence_mse']:>11.6f}"
        )

    if metrics["per_horizon"] and "position" in metrics["per_horizon"][0]:
        print(
            "\nhorizon  oracle-pos  autoregressive-pos  "
            "teacher-forced-pos  persistence-pos"
        )
        for row in metrics["per_horizon"]:
            position = row["position"]
            print(
                f"{row['horizon']:>7}  "
                f"{position['oracle']['mean_euclidean_error']:>10.6f}  "
                f"{position['autoregressive']['mean_euclidean_error']:>18.6f}  "
                f"{position['teacher_forced']['mean_euclidean_error']:>18.6f}  "
                f"{position['persistence']['mean_euclidean_error']:>15.6f}"
            )


def save_metrics_plot(metrics, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = metrics["per_horizon"]
    horizons = [row["horizon"] for row in rows]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        constrained_layout=True,
    )

    latent_axis = axes[0]
    for key, label in [
        ("autoregressive_mse", "Autoregressive"),
        ("teacher_forced_mse", "Teacher-forced"),
        ("persistence_mse", "Persistence"),
    ]:
        latent_axis.plot(
            horizons,
            [row[key] for row in rows],
            marker="o",
            label=label,
        )
    latent_axis.set_xlabel("Rollout horizon")
    latent_axis.set_ylabel("Latent MSE")
    latent_axis.set_title("Latent prediction error")
    latent_axis.legend()
    latent_axis.grid(alpha=0.25)

    position_axis = axes[1]
    for key, label in [
        ("oracle", "Real latent (probe floor)"),
        ("autoregressive", "Autoregressive"),
        ("teacher_forced", "Teacher-forced"),
        ("persistence", "Persistence"),
    ]:
        position_axis.plot(
            horizons,
            [
                row["position"][key]["mean_euclidean_error"]
                for row in rows
            ],
            marker="o",
            label=label,
        )
    position_axis.set_xlabel("Rollout horizon")
    position_axis.set_ylabel("Mean position distance")
    position_axis.set_title("Decoded ball-position error")
    position_axis.legend()
    position_axis.grid(alpha=0.25)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_trajectory_plot(trajectories, output_path):
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
        figsize=(6 * num_columns, 5 * num_rows),
        constrained_layout=True,
        squeeze=False,
    )

    for axis, trajectory in zip(axes.flat, trajectories):
        true_positions = trajectory["true"].numpy()
        oracle_positions = trajectory["oracle"].numpy()
        autoregressive_positions = trajectory["autoregressive"].numpy()
        teacher_forced_positions = trajectory["teacher_forced"].numpy()

        for horizon in range(1, len(true_positions)):
            axis.plot(
                [
                    true_positions[horizon, 0],
                    autoregressive_positions[horizon, 0],
                ],
                [
                    true_positions[horizon, 1],
                    autoregressive_positions[horizon, 1],
                ],
                color="0.75",
                linewidth=0.8,
                zorder=1,
            )

        axis.plot(
            true_positions[:, 0],
            true_positions[:, 1],
            "o-",
            label="True",
            zorder=3,
        )
        axis.plot(
            oracle_positions[:, 0],
            oracle_positions[:, 1],
            "d:",
            label="Real latent decoded (probe floor)",
            alpha=0.85,
            zorder=2,
        )
        axis.plot(
            autoregressive_positions[:, 0],
            autoregressive_positions[:, 1],
            "x-",
            label="Autoregressive",
            zorder=4,
        )
        axis.plot(
            teacher_forced_positions[:, 0],
            teacher_forced_positions[:, 1],
            "^--",
            label="Teacher-forced",
            alpha=0.8,
            zorder=2,
        )
        axis.scatter(
            true_positions[0, 0],
            true_positions[0, 1],
            marker="s",
            s=70,
            label="Start",
            zorder=5,
        )

        for horizon in range(1, len(true_positions)):
            axis.annotate(
                str(horizon),
                true_positions[horizon],
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

        axis.set_xlabel("Maze x")
        axis.set_ylabel("Maze y")
        axis.set_title(
            f"Episode {trajectory['episode_id']:04d}, "
            f"start {trajectory['start']}"
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
        ncol=min(5, len(labels)),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    validate_arguments(args)

    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    model, metadata, checkpoint_epoch = load_model_and_metadata(
        args.checkpoint,
        device,
    )
    model = compile_model(model, enabled=args.compile)

    split_key = f"{args.split}_ids"
    world_model_episode_ids = metadata.get(split_key)
    if world_model_episode_ids is None:
        raise ValueError(
            f"checkpoint metadata does not contain {split_key}"
        )

    position_probe = None
    position_probe_metadata = None
    position_probe_epoch = None
    position_mean = None
    position_std = None
    episode_ids = world_model_episode_ids
    if args.position_probe_checkpoint is not None:
        (
            position_probe,
            position_probe_metadata,
            position_probe_epoch,
        ) = load_probe_checkpoint(
            args.position_probe_checkpoint,
            device,
        )
        probe_episode_ids = position_probe_metadata.get(split_key)
        if probe_episode_ids is None:
            raise ValueError(
                f"position probe metadata does not contain {split_key}"
            )
        episode_ids = intersect_episode_ids(
            world_model_episode_ids,
            probe_episode_ids,
        )
        position_mean = position_probe_metadata["position_mean"]
        position_std = position_probe_metadata["position_std"]

    model_configuration = metadata["model_configuration"]
    num_hist = model_configuration["num_hist"]
    action_mean = None
    action_std = None
    if metadata.get("normalize_actions", False):
        action_mean = metadata.get("action_mean")
        action_std = metadata.get("action_std")
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

    print(f"Checkpoint epoch: {checkpoint_epoch + 1}")
    print(f"Split: {args.split} ({len(episode_ids)} episodes)")
    if position_probe is not None:
        print(
            f"Position probe epoch: {position_probe_epoch + 1} "
            f"({len(position_probe_metadata[split_key])} probe-split episodes)"
        )
    print(f"Rollout samples: {len(dataset)}")
    print(f"History: {num_hist}, horizon: {args.horizon}")
    print(f"Device: {device}, precision: {precision}")

    metrics = evaluate_model(
        model=model,
        data_loader=data_loader,
        device=device,
        num_hist=num_hist,
        precision=precision,
        log_every=args.log_every,
        position_probe=position_probe,
        position_mean=position_mean,
        position_std=position_std,
    )
    metrics.update({
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "split": args.split,
        "num_episodes": len(episode_ids),
        "num_hist": num_hist,
        "max_horizon": args.horizon,
        "episode_ids": episode_ids,
    })
    if position_probe is not None:
        metrics.update({
            "position_probe_checkpoint": str(
                args.position_probe_checkpoint
            ),
            "position_probe_epoch": position_probe_epoch,
            "num_world_model_split_episodes": len(
                world_model_episode_ids
            ),
            "num_probe_split_episodes": len(
                position_probe_metadata[split_key]
            ),
            "num_evaluated_episodes": len(episode_ids),
        })

    output_path = args.output
    if output_path is None:
        suffix = "_position" if position_probe is not None else ""
        output_path = (
            args.checkpoint.parent
            / f"rollout_{args.split}_h{args.horizon}{suffix}.json"
        )

    trajectory_plot_output = None
    trajectories = []
    if position_probe is not None:
        plotted_episode_ids = episode_ids[
            :args.num_trajectory_plots
        ]
        first_sample_index = {
            episode_id: index
            for index, (episode_id, start) in enumerate(dataset.samples)
            if start == 0 and episode_id in plotted_episode_ids
        }
        for episode_id in plotted_episode_ids:
            sample = dataset[first_sample_index[episode_id]]
            trajectories.append(
                predict_position_trajectory(
                    model=model,
                    position_probe=position_probe,
                    sample=sample,
                    device=device,
                    num_hist=num_hist,
                    position_mean=position_mean,
                    position_std=position_std,
                    precision=precision,
                )
            )
        metrics["trajectory_plot_episode_ids"] = plotted_episode_ids
        metrics["trajectory_plot_start"] = 0
        trajectory_plot_output = args.trajectory_plot_output
        if trajectory_plot_output is None:
            trajectory_plot_output = output_path.with_name(
                output_path.stem + "_trajectories.png"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2)

    print_metrics(metrics)
    print(f"\nSaved metrics to: {output_path}")

    if position_probe is not None:
        plot_output = args.plot_output
        if plot_output is None:
            plot_output = output_path.with_suffix(".png")
        save_metrics_plot(metrics, plot_output)
        print(f"Saved diagnostic plot to: {plot_output}")
        save_trajectory_plot(trajectories, trajectory_plot_output)
        print(
            "Saved decoded trajectory plot to: "
            f"{trajectory_plot_output}"
        )


if __name__ == "__main__":
    main()

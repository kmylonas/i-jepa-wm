"""Probe frozen I-JEPA patch embeddings for PointMaze position."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from src.mylonas_ijepa_wm.wm import (
    create_episode_splits,
    resolve_device,
    resolve_num_workers,
    set_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "ijepa_cache"
LATENTS_PATH = DATA_PATH / "latents"
TRAJECTORIES_PATH = DATA_PATH / "trajectories"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Probe frozen I-JEPA tokens for PointMaze position.",
    )
    parser.add_argument("--latent-dir", type=Path, default=LATENTS_PATH)
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=TRAJECTORIES_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "position_probe_runs" / "spatial_softmax",
    )
    parser.add_argument("--num-episodes", type=int, default=2000)
    parser.add_argument("--train-episodes", type=int, default=1600)
    parser.add_argument("--val-episodes", type=int, default=200)
    parser.add_argument("--test-episodes", type=int, default=200)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--ijepa-dim", type=int, default=1280)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--scorer",
        choices=["linear", "mlp"],
        default="linear",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--preload-latents",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from OUTPUT_DIR/last.pt up to the requested epoch.",
    )
    return parser.parse_args(argv)


def create_patch_coordinates(grid_size):
    axis = torch.linspace(-1.0, 1.0, steps=grid_size)
    rows, columns = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((columns, rows), dim=-1).reshape(-1, 2)


def compute_position_statistics(trajectory_dir, episode_ids):
    trajectory_dir = Path(trajectory_dir)
    positions = []

    for episode_id in episode_ids:
        trajectory = torch.load(
            trajectory_dir / f"episode_{episode_id:04d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        positions.append(trajectory["sampled_states"][:, :2].float())

    positions = torch.cat(positions, dim=0)
    position_mean = positions.mean(dim=0)
    position_std = positions.std(dim=0, unbiased=False).clamp_min(1e-6)
    return position_mean, position_std


def compute_position_metrics(predictions, targets, baseline_position):
    predictions = predictions.float()
    targets = targets.float()
    errors = predictions - targets
    squared_errors = errors.square()

    residual_sum = squared_errors.sum(dim=0)
    centered_targets = targets - targets.mean(dim=0)
    total_sum = centered_targets.square().sum(dim=0)
    r_squared = 1.0 - residual_sum / total_sum.clamp_min(1e-12)

    baseline_errors = targets - baseline_position.float().reshape(1, 2)

    return {
        "num_samples": len(targets),
        "mse": squared_errors.mean().item(),
        "mae_x": errors[:, 0].abs().mean().item(),
        "mae_y": errors[:, 1].abs().mean().item(),
        "mean_euclidean_error": errors.norm(dim=1).mean().item(),
        "r2_x": r_squared[0].item(),
        "r2_y": r_squared[1].item(),
        "baseline_mean_euclidean_error": (
            baseline_errors.norm(dim=1).mean().item()
        ),
    }


class SpatialSoftmaxProbe(nn.Module):
    def __init__(
        self,
        ijepa_dim=1280,
        grid_size=16,
        temperature=1.0,
        scorer="linear",
        hidden_dim=128,
    ):
        super().__init__()

        if grid_size < 2:
            raise ValueError("grid-size must be at least 2")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden-dim must be at least 1")

        self.grid_size = grid_size
        self.temperature = temperature
        if scorer == "linear":
            self.patch_scorer = nn.Linear(ijepa_dim, 1)
        elif scorer == "mlp":
            self.patch_scorer = nn.Sequential(
                nn.Linear(ijepa_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(f"Unknown scorer: {scorer}")
        self.coordinate_mapping = nn.Linear(2, 2)
        self.register_buffer(
            "patch_coordinates",
            create_patch_coordinates(grid_size),
        )

    def patch_probabilities(self, tokens):
        scores = self.patch_scorer(tokens).squeeze(-1)
        return torch.softmax(scores / self.temperature, dim=1)

    def forward(self, tokens):
        probabilities = self.patch_probabilities(tokens)
        image_positions = probabilities @ self.patch_coordinates
        return self.coordinate_mapping(image_positions)


class PositionProbeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        latent_dir,
        trajectory_dir,
        episode_ids,
        preload_latents=False,
    ):
        self.latent_dir = Path(latent_dir)
        self.trajectory_dir = Path(trajectory_dir)
        self.episode_ids = list(episode_ids)
        self.preload_latents = preload_latents
        self.positions = {}
        self.latents = {}

        for episode_id in self.episode_ids:
            trajectory = torch.load(
                self.trajectory_dir / f"episode_{episode_id:04d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            positions = trajectory["sampled_states"][:, :2].float()
            latent_path = (
                self.latent_dir / f"episode_{episode_id:04d}.npy"
            )
            latents = np.load(
                latent_path,
                mmap_mode=None if preload_latents else "r",
                allow_pickle=False,
            )

            if len(latents) != len(positions):
                raise ValueError(
                    f"Episode {episode_id} has {len(latents)} latent "
                    f"frames but {len(positions)} sampled states"
                )

            self.positions[episode_id] = positions
            if preload_latents:
                self.latents[episode_id] = latents

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        if self.preload_latents:
            latents = self.latents[episode_id]
        else:
            latents = np.load(
                self.latent_dir / f"episode_{episode_id:04d}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )

        return {
            "latents": torch.from_numpy(np.array(latents, copy=True)),
            "positions": self.positions[episode_id],
            "episode_id": episode_id,
        }


def run_epoch(
    model,
    data_loader,
    device,
    position_mean,
    position_std,
    optimizer=None,
    log_every=0,
):
    is_training = optimizer is not None
    model.train(is_training)

    position_mean = position_mean.to(device=device, dtype=torch.float32)
    position_std = position_std.to(device=device, dtype=torch.float32)
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    start_time = time.perf_counter()
    context = torch.enable_grad() if is_training else torch.inference_mode()

    with context:
        for step, batch in enumerate(data_loader, start=1):
            latents = batch["latents"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=device.type == "cuda",
            )
            positions = batch["positions"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=device.type == "cuda",
            )

            latents = latents.flatten(0, 1)
            positions = positions.flatten(0, 1)
            normalized_positions = (
                positions - position_mean
            ) / position_std

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            normalized_predictions = model(latents)
            loss = torch.nn.functional.mse_loss(
                normalized_predictions,
                normalized_positions,
            )

            if is_training:
                loss.backward()
                optimizer.step()

            predictions = (
                normalized_predictions.detach() * position_std
                + position_mean
            )
            batch_size = len(positions)
            total_loss += loss.detach().item() * batch_size
            total_samples += batch_size
            all_predictions.append(predictions.cpu())
            all_targets.append(positions.detach().cpu())

            if log_every > 0 and step % log_every == 0:
                elapsed = time.perf_counter() - start_time
                print(
                    f"  batch {step}/{len(data_loader)} "
                    f"samples/s={total_samples / elapsed:.2f}"
                )

    if total_samples == 0:
        raise ValueError("data loader is empty")

    metrics = compute_position_metrics(
        predictions=torch.cat(all_predictions),
        targets=torch.cat(all_targets),
        baseline_position=position_mean.cpu(),
    )
    metrics["loss"] = total_loss / total_samples
    return metrics


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    best_val_error,
    history,
    metadata,
):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_error": best_val_error,
            "history": history,
            "metadata": metadata,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def train_model(
    model,
    optimizer,
    train_loader,
    val_loader,
    device,
    position_mean,
    position_std,
    epochs,
    output_dir,
    checkpoint_metadata,
    log_every=0,
    start_epoch=0,
    best_val_error=float("inf"),
    history=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if history is None:
        history = []

    for epoch in range(start_epoch, epochs):
        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            device=device,
            position_mean=position_mean,
            position_std=position_std,
            optimizer=optimizer,
            log_every=log_every,
        )
        val_metrics = run_epoch(
            model=model,
            data_loader=val_loader,
            device=device,
            position_mean=position_mean,
            position_std=position_std,
        )
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        })

        val_error = val_metrics["mean_euclidean_error"]
        if val_error < best_val_error:
            best_val_error = val_error
            save_checkpoint(
                output_dir / "best.pt",
                epoch,
                model,
                optimizer,
                best_val_error,
                history,
                checkpoint_metadata,
            )

        save_checkpoint(
            output_dir / "last.pt",
            epoch,
            model,
            optimizer,
            best_val_error,
            history,
            checkpoint_metadata,
        )

        print(
            f"epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_distance={val_error:.6f} "
            f"val_r2=({val_metrics['r2_x']:.4f}, "
            f"{val_metrics['r2_y']:.4f})"
        )

    return history, best_val_error


def canonical_model_configuration(configuration):
    scorer = configuration.get("scorer", "linear")
    return {
        "ijepa_dim": configuration["ijepa_dim"],
        "grid_size": configuration["grid_size"],
        "temperature": configuration["temperature"],
        "scorer": scorer,
        "hidden_dim": (
            configuration.get("hidden_dim", 128)
            if scorer == "mlp"
            else None
        ),
    }


def load_probe_checkpoint(path, device):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    metadata = checkpoint["metadata"]
    configuration = dict(metadata["model_configuration"])
    configuration.setdefault("scorer", "linear")
    configuration.setdefault("hidden_dim", 128)

    model = SpatialSoftmaxProbe(**configuration)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, metadata, checkpoint.get("epoch")


def load_training_checkpoint(
    path,
    model,
    optimizer,
    current_metadata,
    device,
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )
    saved_metadata = checkpoint["metadata"]

    saved_configuration = canonical_model_configuration(
        saved_metadata["model_configuration"]
    )
    current_configuration = canonical_model_configuration(
        current_metadata["model_configuration"]
    )
    if saved_configuration != current_configuration:
        raise ValueError(
            "Current probe configuration does not match the checkpoint"
        )

    for key in ["train_ids", "val_ids", "test_ids"]:
        if saved_metadata[key] != current_metadata[key]:
            raise ValueError(
                f"Current {key} does not match the checkpoint"
            )
    for key in ["position_mean", "position_std"]:
        if not torch.equal(
            saved_metadata[key].cpu(),
            current_metadata[key].cpu(),
        ):
            raise ValueError(
                f"Current {key} does not match the checkpoint"
            )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return (
        checkpoint["epoch"] + 1,
        checkpoint["best_val_error"],
        checkpoint["history"],
    )


def main(argv=None):
    args = parse_args(argv)

    if not args.latent_dir.is_dir():
        raise NotADirectoryError(
            f"Latent directory not found: {args.latent_dir}"
        )
    if not args.trajectory_dir.is_dir():
        raise NotADirectoryError(
            f"Trajectory directory not found: {args.trajectory_dir}"
        )
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if args.hidden_dim < 1:
        raise ValueError("hidden-dim must be at least 1")

    set_seed(args.seed)
    device = resolve_device(args.device)
    train_ids, val_ids, test_ids = create_episode_splits(
        num_episodes=args.num_episodes,
        train_episodes=args.train_episodes,
        val_episodes=args.val_episodes,
        test_episodes=args.test_episodes,
        seed=args.seed,
    )
    position_mean, position_std = compute_position_statistics(
        args.trajectory_dir,
        train_ids,
    )

    dataset_arguments = {
        "latent_dir": args.latent_dir,
        "trajectory_dir": args.trajectory_dir,
        "preload_latents": args.preload_latents,
    }
    train_dataset = PositionProbeDataset(
        episode_ids=train_ids,
        **dataset_arguments,
    )
    val_dataset = PositionProbeDataset(
        episode_ids=val_ids,
        **dataset_arguments,
    )
    test_dataset = PositionProbeDataset(
        episode_ids=test_ids,
        **dataset_arguments,
    )

    loader_num_workers = resolve_num_workers(
        args.num_workers,
        args.preload_latents,
    )
    if loader_num_workers != args.num_workers:
        print(
            "Preloaded NumPy arrays cannot be shared by spawn-based "
            "workers; using num_workers=0."
        )

    loader_arguments = {
        "batch_size": args.batch_size,
        "num_workers": loader_num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": loader_num_workers > 0,
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_arguments,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        **loader_arguments,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        **loader_arguments,
    )

    model_configuration = {
        "ijepa_dim": args.ijepa_dim,
        "grid_size": args.grid_size,
        "temperature": args.temperature,
        "scorer": args.scorer,
        "hidden_dim": args.hidden_dim,
    }
    model = SpatialSoftmaxProbe(**model_configuration).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    checkpoint_metadata = {
        "model_configuration": model_configuration,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "position_mean": position_mean,
        "position_std": position_std,
    }

    start_epoch = 0
    best_val_error = float("inf")
    history = None
    if args.resume:
        checkpoint_path = args.output_dir / "last.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {checkpoint_path}"
            )
        start_epoch, best_val_error, history = load_training_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            current_metadata=checkpoint_metadata,
            device=device,
        )
        print(f"Resuming from epoch {start_epoch + 1}")

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print(f"Device: {device}")
    print(f"Probe parameters: {parameter_count:,}")
    print(
        f"Episodes: {len(train_ids)} train, {len(val_ids)} val, "
        f"{len(test_ids)} test"
    )
    print(
        f"Training position mean: {position_mean.tolist()} "
        f"std: {position_std.tolist()}"
    )

    train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        position_mean=position_mean,
        position_std=position_std,
        epochs=args.epochs,
        output_dir=args.output_dir,
        checkpoint_metadata=checkpoint_metadata,
        log_every=args.log_every,
        start_epoch=start_epoch,
        best_val_error=best_val_error,
        history=history,
    )

    best_checkpoint = torch.load(
        args.output_dir / "best.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_checkpoint["model"])
    test_metrics = run_epoch(
        model=model,
        data_loader=test_loader,
        device=device,
        position_mean=position_mean,
        position_std=position_std,
    )
    test_metrics["checkpoint_epoch"] = best_checkpoint["epoch"]

    metrics_path = args.output_dir / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as output_file:
        json.dump(test_metrics, output_file, indent=2)

    print("\nBest-checkpoint test metrics")
    print(
        f"Mean distance: {test_metrics['mean_euclidean_error']:.6f} "
        f"(mean baseline: "
        f"{test_metrics['baseline_mean_euclidean_error']:.6f})"
    )
    print(
        f"MAE x/y: {test_metrics['mae_x']:.6f} / "
        f"{test_metrics['mae_y']:.6f}"
    )
    print(
        f"R2 x/y: {test_metrics['r2_x']:.4f} / "
        f"{test_metrics['r2_y']:.4f}"
    )
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()

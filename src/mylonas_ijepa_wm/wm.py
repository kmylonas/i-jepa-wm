import argparse
from collections import OrderedDict
import multiprocessing
from pathlib import Path
import time

import torch
import numpy as np

from src.mylonas_ijepa_wm.vit import ViT


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "ijepa_cache"
LATENTS_PATH = DATA_PATH / "latents"
TRAJECTORIES_PATH = DATA_PATH / "trajectories"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the action-conditioned I-JEPA world model.",
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
        default=REPO_ROOT / "world_model_runs" / "baseline",
    )

    parser.add_argument("--num-episodes", type=int, default=2000)
    parser.add_argument("--train-episodes", type=int, default=1600)
    parser.add_argument("--val-episodes", type=int, default=200)
    parser.add_argument("--test-episodes", type=int, default=200)

    parser.add_argument("--num-hist", type=int, default=3)
    parser.add_argument("--num-patches", type=int, default=256)
    parser.add_argument("--ijepa-dim", type=int, default=1280)
    parser.add_argument("--action-dim", type=int, default=10)
    parser.add_argument("--action-embed-dim", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Total epochs, including completed epochs when resuming.",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "fp32", "bf16"],
        default="auto",
        help="Use BF16 automatically on supported CUDA GPUs.",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the model with torch.compile.",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--preload-latents",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Load each split's latent episodes into system RAM before "
            "training instead of memory-mapping them on demand."
        ),
    )
    parser.add_argument(
        "--normalize-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume at the next epoch using OUTPUT_DIR/last.pt.",
    )

    return parser.parse_args(argv)


def compute_action_statistics(trajectory_dir, episode_ids):
    trajectory_dir = Path(trajectory_dir)
    actions = []

    for episode_id in episode_ids:
        trajectory_path = (
            trajectory_dir / f"episode_{episode_id:04d}.pt"
        )
        trajectory = torch.load(
            trajectory_path,
            map_location="cpu",
            weights_only=True,
        )
        actions.append(trajectory["macro_actions"].float())

    actions = torch.cat(actions, dim=0)
    action_mean = actions.mean(dim=0)
    action_std = actions.std(dim=0, unbiased=False).clamp_min(1e-6)

    return action_mean, action_std


def create_episode_splits(
    num_episodes,
    train_episodes,
    val_episodes,
    test_episodes,
    seed,
):
    if min(train_episodes, val_episodes, test_episodes) <= 0:
        raise ValueError(
            "train, validation, and test episode counts must be positive"
        )
    if train_episodes + val_episodes + test_episodes != num_episodes:
        raise ValueError(
            "train, validation, and test episode counts must "
            "sum to num_episodes"
        )

    generator = torch.Generator().manual_seed(seed)
    episode_ids = torch.randperm(
        num_episodes,
        generator=generator,
    ).tolist()

    train_end = train_episodes
    val_end = train_end + val_episodes

    train_ids = episode_ids[:train_end]
    val_ids = episode_ids[train_end:val_end]
    test_ids = episode_ids[val_end:]

    return train_ids, val_ids, test_ids


class WorldModelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        latent_dir,
        trajectory_dir,
        episode_ids,
        num_hist=3,
        action_mean=None,
        action_std=None,
        episode_cache_size=8,
        preload_latents=False,
    ):
        self.latent_dir = Path(latent_dir)
        self.trajectory_dir = Path(trajectory_dir)
        self.num_hist = num_hist
        self.action_mean = action_mean
        self.action_std = action_std
        self.episode_cache_size = episode_cache_size
        self.preload_latents = preload_latents

        assert (action_mean is None) == (action_std is None)
        if episode_cache_size < 1:
            raise ValueError("episode_cache_size must be at least 1")

        self.samples = []
        self.actions = {}
        self._latent_cache = OrderedDict()

        for episode_id in episode_ids:
            latent_path = (
                self.latent_dir / f"episode_{episode_id:04d}.npy"
            )
            if self.preload_latents:
                latents = np.load(
                    latent_path,
                    allow_pickle=False,
                ).astype(np.float16, copy=False)
                self._latent_cache[episode_id] = latents
            else:
                latents = np.load(
                    latent_path,
                    mmap_mode="r",
                    allow_pickle=False,
                )
            num_frames = len(latents)

            trajectory_path = (
                self.trajectory_dir / f"episode_{episode_id:04d}.pt"
            )
            trajectory = torch.load(
                trajectory_path,
                map_location="cpu",
                weights_only=True,
            )
            actions = trajectory["macro_actions"].float()

            if len(actions) < num_frames - 1:
                raise ValueError(
                    f"Episode {episode_id} has {num_frames} latent frames "
                    f"but only {len(actions)} macro actions"
                )

            self.actions[episode_id] = actions

            for start in range(num_frames - num_hist):
                self.samples.append((episode_id, start))

    def __len__(self):
        return len(self.samples)

    def load_latents(self, episode_id):
        if episode_id in self._latent_cache:
            self._latent_cache.move_to_end(episode_id)
            return self._latent_cache[episode_id]

        latent_path = (
            self.latent_dir / f"episode_{episode_id:04d}.npy"
        )
        latents = np.load(latent_path, mmap_mode="r")
        self._latent_cache[episode_id] = latents

        if len(self._latent_cache) > self.episode_cache_size:
            self._latent_cache.popitem(last=False)

        return latents

    def __getitem__(self, index):
        episode_id, start = self.samples[index]

        latents = self.load_latents(episode_id)

        end = start + self.num_hist

        latent_window = np.array(
            latents[start:end + 1],
            copy=True,
        )

        actions = self.actions[episode_id][start:end]

        if self.action_mean is not None:
            actions = (
                actions - self.action_mean
            ) / self.action_std

        return {
            # (num_hist + 1, 256, 1280)
            "latent_window": torch.from_numpy(latent_window),
            # (num_hist, 5x2=10), where 5 is the frame skip
            "actions": actions,
        }


def run_epoch(
    model,
    data_loader,
    device,
    optimizer=None,
    log_every=0,
    precision="fp32",
):
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = torch.zeros((), device=device)
    total_samples = 0
    loss_fn = torch.nn.MSELoss()
    non_blocking = device.type == "cuda"
    last_log_step = 0
    last_log_time = time.perf_counter()

    context = torch.enable_grad() if is_training else torch.inference_mode()

    with context:
        for step, batch in enumerate(data_loader, start=1):
            latent_window = batch["latent_window"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            )
            input_latents = latent_window[:, :-1]
            target_latents = latent_window[:, 1:]
            actions = batch["actions"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            )

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
            ):
                predicted_latents = model(input_latents, actions)

            loss = loss_fn(
                predicted_latents.float(),
                target_latents,
            )

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = input_latents.shape[0]
            total_loss.add_(loss.detach(), alpha=batch_size)
            total_samples += batch_size

            if is_training and log_every > 0 and step % log_every == 0:
                current_loss = loss.detach().item()
                current_time = time.perf_counter()
                steps_per_second = (
                    (step - last_log_step)
                    / (current_time - last_log_time)
                )
                print(
                    f"  step {step}: loss={current_loss:.6f} "
                    f"steps/s={steps_per_second:.2f}"
                )
                last_log_step = step
                last_log_time = current_time

    if total_samples == 0:
        raise ValueError("data loader is empty")

    return total_loss.item() / total_samples


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    best_val_loss,
    history,
    metadata=None,
):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "history": history,
        "metadata": metadata or {},
    }

    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def load_training_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]
    history = checkpoint["history"]
    metadata = checkpoint.get("metadata", {})

    return start_epoch, best_val_loss, history, metadata


def validate_resume_metadata(saved_metadata, current_metadata):
    exact_keys = [
        "model_configuration",
        "train_ids",
        "val_ids",
        "test_ids",
        "normalize_actions",
    ]

    for key in exact_keys:
        if saved_metadata.get(key) != current_metadata.get(key):
            raise ValueError(
                f"Current {key} does not match the resume checkpoint"
            )

    for key in ["action_mean", "action_std"]:
        saved_value = saved_metadata.get(key)
        current_value = current_metadata.get(key)

        if saved_value is None and current_value is None:
            continue
        if saved_value is None or current_value is None:
            raise ValueError(
                f"Current {key} does not match the resume checkpoint"
            )
        if not torch.equal(
            saved_value.detach().cpu(),
            current_value.detach().cpu(),
        ):
            raise ValueError(
                f"Current {key} does not match the resume checkpoint"
            )


def train_model(
    model,
    optimizer,
    train_loader,
    val_loader,
    device,
    epochs,
    validate_every,
    output_dir,
    log_every=100,
    start_epoch=0,
    best_val_loss=float("inf"),
    history=None,
    checkpoint_metadata=None,
    precision="fp32",
):
    if validate_every < 1:
        raise ValueError("validate_every must be at least 1")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if history is None:
        history = {
            "train_loss": [],
            "val_loss": [],
        }

    for epoch in range(start_epoch, epochs):
        train_loss = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            log_every=log_every,
            precision=precision,
        )
        history["train_loss"].append({
            "epoch": epoch,
            "loss": train_loss,
        })

        should_validate = (
            (epoch + 1) % validate_every == 0
            or epoch == epochs - 1
        )

        val_loss = None
        if should_validate:
            val_loss = run_epoch(
                model,
                val_loader,
                device,
                precision=precision,
            )
            history["val_loss"].append({
                "epoch": epoch,
                "loss": val_loss,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    output_dir / "best.pt",
                    epoch,
                    model,
                    optimizer,
                    best_val_loss,
                    history,
                    checkpoint_metadata,
                )

        save_checkpoint(
            output_dir / "last.pt",
            epoch,
            model,
            optimizer,
            best_val_loss,
            history,
            checkpoint_metadata,
        )

        message = f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.6f}"
        if val_loss is not None:
            message += f" val_loss={val_loss:.6f}"
        print(message)

    return history, best_val_loss


def resolve_device(device_name):
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")

    return torch.device(device_name)


def resolve_precision(precision, device, bf16_supported=None):
    if bf16_supported is None:
        bf16_supported = (
            device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )

    if precision == "auto":
        return "bf16" if bf16_supported else "fp32"

    if precision == "bf16" and not bf16_supported:
        raise ValueError(
            "BF16 precision requires a CUDA GPU with BF16 support"
        )

    return precision


def compile_model(model, enabled, backend=None):
    if enabled:
        compile_arguments = {}
        if backend is not None:
            compile_arguments["backend"] = backend
        model.compile(**compile_arguments)

    return model


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_num_workers(
    num_workers,
    preload_latents,
    start_method=None,
):
    if num_workers == 0 or not preload_latents:
        return num_workers

    if start_method is None:
        start_method = multiprocessing.get_start_method()

    if start_method != "fork":
        return 0

    return num_workers


def create_data_loaders(
    args,
    train_ids,
    val_ids,
    test_ids,
    action_mean,
    action_std,
):
    dataset_arguments = {
        "latent_dir": args.latent_dir,
        "trajectory_dir": args.trajectory_dir,
        "num_hist": args.num_hist,
        "action_mean": action_mean,
        "action_std": action_std,
        "preload_latents": args.preload_latents,
    }

    train_dataset = WorldModelDataset(
        episode_ids=train_ids,
        **dataset_arguments,
    )
    val_dataset = WorldModelDataset(
        episode_ids=val_ids,
        **dataset_arguments,
    )
    test_dataset = WorldModelDataset(
        episode_ids=test_ids,
        **dataset_arguments,
    )

    generator = torch.Generator().manual_seed(args.seed)
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
        "pin_memory": args.device == "cuda",
        "persistent_workers": loader_num_workers > 0,
    }

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

    return train_loader, val_loader, test_loader


def validate_arguments(args):
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
    if args.validate_every < 1:
        raise ValueError("validate-every must be at least 1")


def main(argv=None):
    args = parse_args(argv)
    validate_arguments(args)
    set_seed(args.seed)

    device = resolve_device(args.device)
    args.device = device.type
    precision = resolve_precision(args.precision, device)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    train_ids, val_ids, test_ids = create_episode_splits(
        num_episodes=args.num_episodes,
        train_episodes=args.train_episodes,
        val_episodes=args.val_episodes,
        test_episodes=args.test_episodes,
        seed=args.seed,
    )

    action_mean = None
    action_std = None
    if args.normalize_actions:
        action_mean, action_std = compute_action_statistics(
            args.trajectory_dir,
            train_ids,
        )

    train_loader, val_loader, test_loader = create_data_loaders(
        args,
        train_ids,
        val_ids,
        test_ids,
        action_mean,
        action_std,
    )

    model_configuration = {
        "num_patches": args.num_patches,
        "num_hist": args.num_hist,
        "num_t_blocks": args.num_blocks,
        "ijepa_dim": args.ijepa_dim,
        "action_dim": args.action_dim,
        "action_embed_dim": args.action_embed_dim,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "mlp_dim": args.mlp_dim,
        "dropout": args.dropout,
    }

    model = ViT(**model_configuration).to(device)
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
        "normalize_actions": args.normalize_actions,
        "action_mean": action_mean,
        "action_std": action_std,
    }

    start_epoch = 0
    best_val_loss = float("inf")
    history = None

    if args.resume:
        checkpoint_path = args.output_dir / "last.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {checkpoint_path}"
            )

        (
            start_epoch,
            best_val_loss,
            history,
            saved_metadata,
        ) = load_training_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            device,
        )

        validate_resume_metadata(
            saved_metadata,
            checkpoint_metadata,
        )
        print(f"Resuming from epoch {start_epoch + 1}")

    model = compile_model(model, enabled=args.compile)

    print(f"Using device: {device}")
    print(f"Using precision: {precision}")
    print(f"Model compilation: {args.compile}")
    print(f"Preloaded latents: {args.preload_latents}")
    print(
        "Samples: "
        f"train={len(train_loader.dataset)}, "
        f"validation={len(val_loader.dataset)}, "
        f"test={len(test_loader.dataset)}"
    )

    history, best_val_loss = train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        validate_every=args.validate_every,
        output_dir=args.output_dir,
        log_every=args.log_every,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        history=history,
        checkpoint_metadata=checkpoint_metadata,
        precision=precision,
    )

    best_checkpoint_path = args.output_dir / "best.pt"
    if not best_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Best checkpoint not found: {best_checkpoint_path}"
        )

    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_checkpoint["model"])
    test_loss = run_epoch(
        model,
        test_loader,
        device,
        precision=precision,
    )

    torch.save(
        {
            "test_loss": test_loss,
            "best_epoch": best_checkpoint["epoch"],
            "best_val_loss": best_val_loss,
        },
        args.output_dir / "test_metrics.pt",
    )

    print(
        f"best_epoch={best_checkpoint['epoch'] + 1} "
        f"best_val_loss={best_val_loss:.6f} "
        f"test_loss={test_loss:.6f}"
    )


if __name__ == "__main__":
    main()

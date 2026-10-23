"""Train a top-down I-JEPA world model conditioned on position and velocity."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.mylonas_ijepa_wm import wm
from src.mylonas_ijepa_wm.proprio_vit import ProprioViT
from src.mylonas_ijepa_wm.environment_configs import load_environment_config


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = MODULE_DIR / "topdown" / "ijepa_cache"
DEFAULT_OUTPUT = MODULE_DIR / "topdown" / "world_models" / "proprio_v1"
MODEL_TYPE = "proprio_v1"
PROPRIO_FIELDS = ("x", "y", "vx", "vy")


def read_physics_frame_skip(trajectory_dir, episode_ids):
    """Read episode timing and reject caches that mix different dynamics."""
    expected = None
    for episode_id in episode_ids:
        trajectory = torch.load(
            Path(trajectory_dir) / f"episode_{episode_id:04d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        value = int(trajectory.get("physics_frame_skip", 1))
        if value < 1:
            raise ValueError(f"Episode {episode_id} has invalid physics_frame_skip")
        if expected is not None and value != expected:
            raise ValueError(
                f"Episode {episode_id} physics_frame_skip={value} does not "
                f"match the rest of the cache ({expected})"
            )
        expected = value
    if expected is None:
        raise ValueError("episode_ids must not be empty")
    return expected


def read_checkpoint_physics_frame_skip(checkpoint_path):
    """Read timing without constructing the model or moving it to a GPU."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    value = int(checkpoint.get("metadata", {}).get("physics_frame_skip", 1))
    if value < 1:
        raise ValueError("checkpoint physics_frame_skip must be positive")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a top-down I-JEPA world model with proprioception."
    )
    parser.add_argument("--latent-dir", type=Path, default=DEFAULT_CACHE / "latents")
    parser.add_argument("--env", choices=["pointmaze", "wall"], default="pointmaze")
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_CACHE / "trajectories")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-episodes", type=int, default=2000)
    parser.add_argument("--train-episodes", type=int, default=1600)
    parser.add_argument("--val-episodes", type=int, default=200)
    parser.add_argument("--test-episodes", type=int, default=200)
    parser.add_argument("--num-hist", type=int, default=3)
    parser.add_argument("--num-patches", type=int, default=256)
    parser.add_argument("--ijepa-dim", type=int, default=1280)
    parser.add_argument("--action-dim", type=int)
    parser.add_argument("--action-embed-dim", type=int, default=32)
    parser.add_argument("--proprio-embed-dim", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument("--rollout-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--proprio-loss-weight",
        type=float,
        default=1.0,
        help="Weight on MSE of normalized next position and velocity.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preload-latents", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--normalize-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=100)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", action="store_true")
    checkpoint_group.add_argument("--finetune-from", type=Path)
    return parser.parse_args(argv)


def compute_proprio_statistics(trajectory_dir, episode_ids, state_dim=4):
    states = []
    for episode_id in episode_ids:
        trajectory = torch.load(
            Path(trajectory_dir) / f"episode_{episode_id:04d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        sampled_states = trajectory["sampled_states"].float()
        if sampled_states.ndim != 2 or sampled_states.shape[1] != state_dim:
            raise ValueError(f"Episode {episode_id} sampled_states must be [frames, {state_dim}]")
        if not torch.isfinite(sampled_states).all():
            raise ValueError(f"Episode {episode_id} sampled_states are not finite")
        states.append(sampled_states)
    all_states = torch.cat(states, dim=0)
    return (
        all_states.mean(dim=0),
        all_states.std(dim=0, unbiased=False).clamp_min(1e-6),
    )


class ProprioWorldModelDataset(wm.WorldModelDataset):
    def __init__(
        self,
        latent_dir,
        trajectory_dir,
        episode_ids,
        num_hist=3,
        action_mean=None,
        action_std=None,
        state_mean=None,
        state_std=None,
        episode_cache_size=8,
        preload_latents=False,
        rollout_steps=1,
        state_dim=4,
    ):
        super().__init__(
            latent_dir=latent_dir,
            trajectory_dir=trajectory_dir,
            episode_ids=episode_ids,
            num_hist=num_hist,
            action_mean=action_mean,
            action_std=action_std,
            episode_cache_size=episode_cache_size,
            preload_latents=preload_latents,
            rollout_steps=rollout_steps,
        )
        self.state_mean = torch.zeros(state_dim) if state_mean is None else state_mean.float()
        self.state_std = torch.ones(state_dim) if state_std is None else state_std.float()
        if self.state_mean.shape != (state_dim,) or self.state_std.shape != (state_dim,):
            raise ValueError(f"state normalization must have {state_dim} components")
        if not torch.isfinite(self.state_mean).all() or not torch.isfinite(self.state_std).all():
            raise ValueError("state normalization must be finite")
        if not torch.all(self.state_std > 0):
            raise ValueError("state_std must be positive")

        self.states = {}
        for episode_id in episode_ids:
            trajectory = torch.load(
                self.trajectory_dir / f"episode_{episode_id:04d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            if "sampled_states" not in trajectory:
                raise ValueError(f"Episode {episode_id} has no sampled_states")
            states = trajectory["sampled_states"].float()
            num_frames = len(np.load(
                self.latent_dir / f"episode_{episode_id:04d}.npy",
                mmap_mode="r",
                allow_pickle=False,
            ))
            if states.shape != (num_frames, state_dim):
                raise ValueError(
                    f"Episode {episode_id} sampled_states must have shape [{num_frames}, {state_dim}]"
                )
            if not torch.isfinite(states).all():
                raise ValueError(f"Episode {episode_id} sampled_states are not finite")
            self.states[episode_id] = (states - self.state_mean) / self.state_std

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        episode_id, start = self.samples[index]
        end = start + self.num_hist + self.rollout_steps
        sample["state_window"] = self.states[episode_id][start:end]
        return sample


def autoregressive_rollout(model, latent_history, state_history, actions):
    """Roll out predicted visual tokens and normalized proprioceptive state."""
    num_hist = latent_history.shape[1]
    horizon = actions.shape[1] - num_hist + 1
    if horizon < 1 or state_history.shape[:2] != latent_history.shape[:2]:
        raise ValueError("latent, state, and action histories are not aligned")
    visual_predictions = []
    state_predictions = []
    for step in range(horizon):
        visual_window, state_window = model(
            latent_history,
            state_history,
            actions[:, step:step + num_hist],
        )
        next_visual = visual_window[:, -1]
        next_state = state_window[:, -1]
        visual_predictions.append(next_visual)
        state_predictions.append(next_state)
        latent_history = torch.cat(
            [latent_history[:, 1:], next_visual.unsqueeze(1)], dim=1
        )
        state_history = torch.cat(
            [state_history[:, 1:], next_state.unsqueeze(1)], dim=1
        )
    return (
        torch.stack(visual_predictions, dim=1),
        torch.stack(state_predictions, dim=1),
    )


def compute_world_model_loss(
    model,
    latent_window,
    state_window,
    actions,
    num_hist,
    rollout_steps=1,
    rollout_loss_weight=0.0,
    proprio_loss_weight=1.0,
):
    if rollout_steps < 1 or rollout_loss_weight < 0 or proprio_loss_weight <= 0:
        raise ValueError("invalid rollout or proprio loss weight")
    expected_frames = num_hist + rollout_steps
    if latent_window.shape[1] != expected_frames or state_window.shape[:2] != latent_window.shape[:2]:
        raise ValueError("latent and state windows are not aligned")
    if actions.shape[1] != expected_frames - 1:
        raise ValueError("action window is not aligned")

    visual_pred, state_pred = model(
        latent_window[:, :num_hist],
        state_window[:, :num_hist],
        actions[:, :num_hist],
    )
    mse = torch.nn.functional.mse_loss
    one_visual = mse(visual_pred.float(), latent_window[:, 1:num_hist + 1])
    one_state = mse(state_pred.float(), state_window[:, 1:num_hist + 1])
    next_visual = visual_pred[:, -1]
    next_state = state_pred[:, -1]
    visual_predictions = [next_visual]
    state_predictions = [next_state]
    latent_history = torch.cat(
        [latent_window[:, 1:num_hist], next_visual.unsqueeze(1)], dim=1
    )
    state_history = torch.cat(
        [state_window[:, 1:num_hist], next_state.unsqueeze(1)], dim=1
    )
    for horizon in range(1, rollout_steps):
        predicted_visual, predicted_state = model(
            latent_history,
            state_history,
            actions[:, horizon:horizon + num_hist],
        )
        next_visual = predicted_visual[:, -1]
        next_state = predicted_state[:, -1]
        visual_predictions.append(next_visual)
        state_predictions.append(next_state)
        latent_history = torch.cat(
            [latent_history[:, 1:], next_visual.unsqueeze(1)], dim=1
        )
        state_history = torch.cat(
            [state_history[:, 1:], next_state.unsqueeze(1)], dim=1
        )
    visual_rollout = torch.stack(visual_predictions, dim=1)
    state_rollout = torch.stack(state_predictions, dim=1)
    rollout_visual = mse(
        visual_rollout.float(), latent_window[:, num_hist:expected_frames]
    )
    rollout_state = mse(
        state_rollout.float(), state_window[:, num_hist:expected_frames]
    )
    total = (
        one_visual + proprio_loss_weight * one_state
        + rollout_loss_weight * (
            rollout_visual + proprio_loss_weight * rollout_state
        )
    )
    return {
        "loss": total,
        "one_step_visual_loss": one_visual,
        "one_step_proprio_loss": one_state,
        "rollout_visual_loss": rollout_visual,
        "rollout_proprio_loss": rollout_state,
    }


def run_epoch(
    model,
    data_loader,
    device,
    optimizer=None,
    precision="fp32",
    rollout_steps=1,
    rollout_loss_weight=0.0,
    proprio_loss_weight=1.0,
    log_every=0,
):
    model.train(optimizer is not None)
    totals = {}
    total_samples = 0
    context = torch.enable_grad() if optimizer is not None else torch.inference_mode()
    with context:
        for step, batch in enumerate(data_loader, start=1):
            latent_window = batch["latent_window"].to(device=device, dtype=torch.float32)
            state_window = batch["state_window"].to(device=device, dtype=torch.float32)
            actions = batch["actions"].to(device=device, dtype=torch.float32)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
            ):
                metrics = compute_world_model_loss(
                    model,
                    latent_window,
                    state_window,
                    actions,
                    num_hist=latent_window.shape[1] - rollout_steps,
                    rollout_steps=rollout_steps,
                    rollout_loss_weight=rollout_loss_weight,
                    proprio_loss_weight=proprio_loss_weight,
                )
            if optimizer is not None:
                metrics["loss"].backward()
                optimizer.step()
            batch_size = latent_window.shape[0]
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
            total_samples += batch_size
            if optimizer is not None and log_every and step % log_every == 0:
                print(f"  step {step}: loss={float(metrics['loss'].detach()):.6f}")
    if not total_samples:
        raise ValueError("data loader is empty")
    return {name: value / total_samples for name, value in totals.items()}


def validate_checkpoint_metadata(saved, current, include_objective=True):
    wm.validate_checkpoint_metadata(saved, current, include_objective)
    if saved.get("environment", "pointmaze") != current.get("environment", "pointmaze"):
        raise ValueError("Current environment does not match the checkpoint")
    if int(saved.get("actions_per_model_step", 5)) != current.get("actions_per_model_step", 5):
        raise ValueError("Current action grouping does not match the checkpoint")
    if saved.get("environment_config") and saved["environment_config"] != current.get("environment_config"):
        raise ValueError("Current environment configuration does not match the checkpoint")
    if int(saved.get("physics_frame_skip", 1)) != current["physics_frame_skip"]:
        raise ValueError("Current physics_frame_skip does not match the checkpoint")
    for key in ("model_type", "camera", "proprio_fields", "proprio_loss_weight"):
        if saved.get(key) != current.get(key):
            raise ValueError(f"Current {key} does not match the checkpoint")
    for key in ("state_mean", "state_std"):
        if not torch.equal(saved[key].cpu(), current[key].cpu()):
            raise ValueError(f"Current {key} does not match the checkpoint")


def load_model_and_metadata(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("metadata", {})
    if metadata.get("model_type") != MODEL_TYPE or metadata.get("camera") != "topdown":
        raise ValueError("checkpoint is not a top-down proprioceptive world model")
    env_name = metadata.get("environment", "pointmaze")
    expected_fields = load_environment_config(env_name).state_fields
    if tuple(metadata.get("proprio_fields", ())) != expected_fields:
        raise ValueError("checkpoint proprioception fields do not match its environment")
    for key in ("state_mean", "state_std"):
        if key not in metadata or metadata[key].shape != (len(expected_fields),):
            raise ValueError(f"checkpoint has invalid {key}")
    model = ProprioViT(**metadata["model_configuration"])
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, metadata, checkpoint["epoch"]


def main(argv=None):
    args = parse_args(argv)
    config = load_environment_config(args.env, args.env_config)
    if not args.latent_dir.is_dir() or not args.trajectory_dir.is_dir():
        raise NotADirectoryError("top-down latent and trajectory directories are required")
    if args.epochs < 1 or args.batch_size < 1 or args.validate_every < 1:
        raise ValueError("epochs, batch-size, and validate-every must be positive")
    if args.rollout_steps < 1 or args.rollout_loss_weight < 0 or args.proprio_loss_weight <= 0:
        raise ValueError("invalid rollout or proprio loss weight")
    if args.finetune_from is not None and not args.finetune_from.is_file():
        raise FileNotFoundError(args.finetune_from)

    wm.set_seed(args.seed)
    device = wm.resolve_device(args.device)
    precision = wm.resolve_precision(args.precision, device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    train_ids, val_ids, test_ids = wm.create_episode_splits(
        args.num_episodes,
        args.train_episodes,
        args.val_episodes,
        args.test_episodes,
        args.seed,
    )
    physics_frame_skip = read_physics_frame_skip(
        args.trajectory_dir, [*train_ids, *val_ids, *test_ids]
    )
    if args.env == "pointmaze" and args.env_config is None:
        first = torch.load(
            args.trajectory_dir / f"episode_{train_ids[0]:04d}.pt",
            map_location="cpu", weights_only=True,
        )
        config = replace(
            config,
            pointmaze_physics_frame_skip=physics_frame_skip,
            actions_per_model_step=int(first.get("actions_per_model_step", 5)),
        )
    state_dim = len(config.state_fields)
    action_dim = config.macro_action_dim if args.action_dim is None else args.action_dim
    if action_dim != config.macro_action_dim:
        raise ValueError(f"--action-dim must be {config.macro_action_dim} for {args.env}")
    expected_physics = config.pointmaze_physics_frame_skip if args.env == "pointmaze" else 1
    if physics_frame_skip != expected_physics:
        raise ValueError("Cache physics_frame_skip does not match --env-config")
    for episode_id in [*train_ids, *val_ids, *test_ids]:
        trajectory = torch.load(
            args.trajectory_dir / f"episode_{episode_id:04d}.pt",
            map_location="cpu", weights_only=True,
        )
        if trajectory.get("environment", "pointmaze") != args.env:
            raise ValueError(f"Episode {episode_id} belongs to another environment")
        if trajectory["macro_actions"].shape[-1] != config.macro_action_dim:
            raise ValueError(f"Episode {episode_id} has incompatible actions")
        if trajectory["sampled_states"].shape[-1] != state_dim:
            raise ValueError(f"Episode {episode_id} has incompatible state dimension")
        if tuple(trajectory.get("state_fields", PROPRIO_FIELDS)) != config.state_fields:
            raise ValueError(f"Episode {episode_id} has incompatible state fields")
        if trajectory.get("environment_config") and trajectory["environment_config"] != config.portable_metadata():
            raise ValueError(f"Episode {episode_id} has different environment settings")
    action_mean = action_std = None
    if args.normalize_actions:
        action_mean, action_std = wm.compute_action_statistics(
            args.trajectory_dir, train_ids
        )
    state_mean, state_std = compute_proprio_statistics(
        args.trajectory_dir, train_ids, state_dim=state_dim
    )
    dataset_args = dict(
        latent_dir=args.latent_dir,
        trajectory_dir=args.trajectory_dir,
        num_hist=args.num_hist,
        action_mean=action_mean,
        action_std=action_std,
        state_mean=state_mean,
        state_std=state_std,
        preload_latents=args.preload_latents,
        rollout_steps=args.rollout_steps,
        state_dim=state_dim,
    )
    train_dataset = ProprioWorldModelDataset(episode_ids=train_ids, **dataset_args)
    val_dataset = ProprioWorldModelDataset(episode_ids=val_ids, **dataset_args)
    test_dataset = ProprioWorldModelDataset(episode_ids=test_ids, **dataset_args)
    loader_workers = wm.resolve_num_workers(args.num_workers, args.preload_latents)
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=loader_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=loader_workers > 0,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_args,
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **loader_args)

    model_configuration = dict(
        num_patches=args.num_patches,
        num_hist=args.num_hist,
        num_t_blocks=args.num_blocks,
        ijepa_dim=args.ijepa_dim,
        action_dim=action_dim,
        action_embed_dim=args.action_embed_dim,
        proprio_dim=state_dim,
        proprio_embed_dim=args.proprio_embed_dim,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
    )
    model = ProprioViT(**model_configuration).to(device)
    metadata = dict(
        model_type=MODEL_TYPE,
        environment=args.env,
        environment_config=config.portable_metadata(),
        actions_per_model_step=config.actions_per_model_step,
        camera="topdown",
        physics_frame_skip=physics_frame_skip,
        proprio_fields=config.state_fields,
        model_configuration=model_configuration,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        normalize_actions=args.normalize_actions,
        action_mean=action_mean,
        action_std=action_std,
        state_mean=state_mean,
        state_std=state_std,
        proprio_loss_weight=args.proprio_loss_weight,
        training_objective=dict(
            rollout_steps=args.rollout_steps,
            rollout_loss_weight=args.rollout_loss_weight,
        ),
    )
    if args.finetune_from is not None:
        saved = wm.load_model_weights(args.finetune_from, model, device)
        validate_checkpoint_metadata(saved, metadata, include_objective=False)
        metadata["finetune_from"] = str(args.finetune_from)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_epoch = 0
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    if args.resume:
        checkpoint_path = args.output_dir / "last.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        start_epoch, best_val_loss, history, saved = wm.load_training_checkpoint(
            checkpoint_path, model, optimizer, device
        )
        validate_checkpoint_metadata(saved, metadata)
    model = wm.compile_model(model, enabled=args.compile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}, precision: {precision}, camera: topdown")
    print(f"Samples: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(
            model, train_loader, device,
            optimizer=optimizer,
            precision=precision,
            rollout_steps=args.rollout_steps,
            rollout_loss_weight=args.rollout_loss_weight,
            proprio_loss_weight=args.proprio_loss_weight,
            log_every=args.log_every,
        )
        history["train_loss"].append({"epoch": epoch, **train_metrics})
        val_metrics = None
        if (epoch + 1) % args.validate_every == 0 or epoch == args.epochs - 1:
            val_metrics = run_epoch(
                model, val_loader, device,
                precision=precision,
                rollout_steps=args.rollout_steps,
                rollout_loss_weight=args.rollout_loss_weight,
                proprio_loss_weight=args.proprio_loss_weight,
            )
            history["val_loss"].append({"epoch": epoch, **val_metrics})
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                wm.save_checkpoint(
                    args.output_dir / "best.pt", epoch, model, optimizer,
                    best_val_loss, history, metadata,
                )
        wm.save_checkpoint(
            args.output_dir / "last.pt", epoch, model, optimizer,
            best_val_loss, history, metadata,
        )
        val_text = "" if val_metrics is None else f" val={val_metrics['loss']:.6f}"
        print(f"epoch {epoch + 1}/{args.epochs}: train={train_metrics['loss']:.6f}{val_text}")

    best_path = args.output_dir / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(best_path)
    best_model, _, best_epoch = load_model_and_metadata(best_path, device)
    test_metrics = run_epoch(
        best_model, test_loader, device,
        precision=precision,
        rollout_steps=args.rollout_steps,
        rollout_loss_weight=args.rollout_loss_weight,
        proprio_loss_weight=args.proprio_loss_weight,
    )
    torch.save(
        {"best_epoch": best_epoch, "best_val_loss": best_val_loss, **test_metrics},
        args.output_dir / "test_metrics.pt",
    )
    print(f"best_epoch={best_epoch + 1} test_loss={test_metrics['loss']:.6f}")


if __name__ == "__main__":
    main()

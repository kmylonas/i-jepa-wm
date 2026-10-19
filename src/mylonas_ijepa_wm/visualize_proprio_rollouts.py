"""Plot real and predicted PointMaze states from proprioceptive rollouts."""

import argparse
import json
from pathlib import Path

import torch

from src.mylonas_ijepa_wm import evaluate_rollouts_proprio, wm, wm_proprio


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
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num-episodes", type=int, default=4)
    parser.add_argument(
        "--episode-ids", type=int, nargs="+",
        help="Specific checkpoint split episodes; otherwise use the first episodes.",
    )
    parser.add_argument(
        "--velocity-arrow-scale", type=float, default=0.04,
        help="Maze coordinate units per velocity unit in plotted arrows.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def predict_state_trajectory(
    model, sample, num_hist, episode_id, state_mean, state_std, device,
    precision="fp32", start=0,
):
    latents = sample["latent_window"].unsqueeze(0).to(device=device, dtype=torch.float32)
    states = sample["state_window"].unsqueeze(0).to(device=device, dtype=torch.float32)
    actions = sample["actions"].unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16",
        ):
            _, autoregressive_states = wm_proprio.autoregressive_rollout(
                model, latents[:, :num_hist], states[:, :num_hist], actions,
            )
            _, teacher_forced_states = evaluate_rollouts_proprio.teacher_forced_rollout(
                model, latents, states, actions, num_hist,
            )

    initial_state = states[:, num_hist - 1:num_hist]
    mean = torch.as_tensor(state_mean, dtype=torch.float32)
    std = torch.as_tensor(state_std, dtype=torch.float32)

    def to_maze_units(normalized):
        return normalized[0].float().cpu() * std + mean

    return {
        "episode_id": int(episode_id),
        "start": int(start),
        "true": to_maze_units(states[:, num_hist - 1:]),
        "autoregressive": to_maze_units(torch.cat(
            (initial_state, autoregressive_states), dim=1,
        )),
        "teacher_forced": to_maze_units(torch.cat(
            (initial_state, teacher_forced_states), dim=1,
        )),
    }


def save_trajectory_plot(trajectories, output_path, velocity_arrow_scale=0.04):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not trajectories:
        raise ValueError("at least one trajectory is required")
    if velocity_arrow_scale <= 0:
        raise ValueError("velocity_arrow_scale must be positive")

    num_columns = min(2, len(trajectories))
    num_rows = (len(trajectories) + num_columns - 1) // num_columns
    figure, axes = plt.subplots(
        num_rows, num_columns,
        figsize=(7 * num_columns, 6 * num_rows),
        constrained_layout=True,
        squeeze=False,
    )
    styles = (
        ("true", "True", "tab:blue", "o-"),
        ("autoregressive", "Autoregressive", "tab:green", "x-"),
        ("teacher_forced", "Teacher-forced", "tab:red", "^--"),
    )
    for axis, trajectory in zip(axes.flat, trajectories):
        horizon = len(trajectory["true"]) - 1
        arrow_steps = sorted({max(1, horizon // 2), horizon})
        true_states = trajectory["true"].numpy()
        autoregressive_states = trajectory["autoregressive"].numpy()
        for step in arrow_steps:
            axis.plot(
                [true_states[step, 0], autoregressive_states[step, 0]],
                [true_states[step, 1], autoregressive_states[step, 1]],
                color="0.8", linewidth=1, zorder=1,
            )

        for name, label, color, line_style in styles:
            states = trajectory[name].numpy()
            axis.plot(
                states[:, 0], states[:, 1], line_style,
                color=color, label=label, linewidth=1.8,
                markersize=5, zorder=3,
            )
            for step in arrow_steps:
                axis.quiver(
                    states[step, 0], states[step, 1],
                    states[step, 2] * velocity_arrow_scale,
                    states[step, 3] * velocity_arrow_scale,
                    angles="xy", scale_units="xy", scale=1,
                    color=color, width=0.004, alpha=0.75,
                    zorder=4,
                )

        axis.scatter(
            true_states[0, 0], true_states[0, 1],
            marker="s", s=75, color="tab:blue", label="Observed start", zorder=5,
        )
        for step in arrow_steps:
            axis.annotate(
                str(step), true_states[step, :2],
                xytext=(5, 5), textcoords="offset points", fontsize=9,
            )
        axis.set_title(f"Episode {trajectory['episode_id']:04d}, start {trajectory['start']}")
        axis.set_xlabel("Maze x")
        axis.set_ylabel("Maze y")
        axis.set_aspect("equal", adjustable="datalim")
        axis.margins(0.15)
        axis.grid(alpha=0.25)

    for axis in axes.flat[len(trajectories):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=len(labels))
    figure.supxlabel(
        "Velocity arrows at middle and final horizon; "
        f"1 velocity unit = {velocity_arrow_scale:g} maze units of arrow length"
    )
    output_path = Path(output_path)
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
    if args.horizon < 1 or args.num_episodes < 1 or args.velocity_arrow_scale <= 0:
        raise ValueError("horizon, num-episodes, and velocity-arrow-scale must be positive")

    device = wm.resolve_device(args.device)
    precision = wm.resolve_precision(args.precision, device)
    model, metadata, epoch = wm_proprio.load_model_and_metadata(args.checkpoint, device)
    split_ids = metadata.get(f"{args.split}_ids")
    if not split_ids:
        raise ValueError(f"checkpoint has no {args.split}_ids")
    if args.episode_ids is None:
        episode_ids = split_ids[:args.num_episodes]
    else:
        episode_ids = args.episode_ids
        if len(set(episode_ids)) != len(episode_ids) or not set(episode_ids).issubset(split_ids):
            raise ValueError("episode-ids must be distinct members of the checkpoint split")

    if metadata.get("normalize_actions", False):
        action_mean = metadata.get("action_mean")
        action_std = metadata.get("action_std")
        if action_mean is None or action_std is None:
            raise ValueError("normalized-action checkpoint is missing action statistics")
    else:
        action_mean = action_std = None
    num_hist = model.num_hist
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
    first_window = {
        episode_id: index
        for index, (episode_id, start) in enumerate(dataset.samples)
        if start == 0
    }
    if any(episode_id not in first_window for episode_id in episode_ids):
        raise ValueError("horizon is longer than the cached episode permits")

    trajectories = [
        predict_state_trajectory(
            model=model,
            sample=dataset[first_window[episode_id]],
            num_hist=num_hist,
            episode_id=episode_id,
            state_mean=metadata["state_mean"],
            state_std=metadata["state_std"],
            device=device,
            precision=precision,
        )
        for episode_id in episode_ids
    ]
    output = args.output or args.checkpoint.parent / f"rollout_{args.split}_h{args.horizon}_state_trajectories.png"
    save_trajectory_plot(trajectories, output, args.velocity_arrow_scale)
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": epoch,
        "split": args.split,
        "horizon": args.horizon,
        "velocity_arrow_scale": args.velocity_arrow_scale,
        "trajectories": [
            {
                key: value.tolist() if isinstance(value, torch.Tensor) else value
                for key, value in trajectory.items()
            }
            for trajectory in trajectories
        ],
    }, indent=2), encoding="utf-8")
    print(f"Checkpoint epoch: {epoch + 1}; split: {args.split}; episodes: {episode_ids}")
    print(f"Saved trajectory plot to: {output}")
    print(f"Saved trajectory states to: {sidecar}")


if __name__ == "__main__":
    main()

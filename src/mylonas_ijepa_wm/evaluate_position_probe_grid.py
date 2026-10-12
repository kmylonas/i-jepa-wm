"""Map PointMaze position-probe error across the visible maze."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.mylonas_ijepa_wm.embed_frames import preprocess_frames
from src.mylonas_ijepa_wm.evaluate_rollouts import decode_positions
from src.mylonas_ijepa_wm.plan_pointmaze import (
    DEFAULT_ENCODER_CHECKPOINT,
    DEFAULT_POSITION_PROBE_CHECKPOINT,
    create_environment,
    extract_wall_rectangles,
    hide_goal_marker,
    load_frozen_ijepa_encoder,
    validate_frame,
)
from src.mylonas_ijepa_wm.position_probe import load_probe_checkpoint
from src.mylonas_ijepa_wm.wm import resolve_device, resolve_precision


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = MODULE_DIR / "position_probe_grid_runs" / "angled_camera"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PointMaze position-probe error on a dense spatial "
            "grid rendered with the environment camera."
        ),
    )
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        default=DEFAULT_ENCODER_CHECKPOINT,
    )
    parser.add_argument(
        "--position-probe-checkpoint",
        type=Path,
        default=DEFAULT_POSITION_PROBE_CHECKPOINT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--environment-id",
        default="PointMaze_UMaze-v3",
    )
    parser.add_argument("--grid-resolution", type=int, default=61)
    parser.add_argument(
        "--wall-clearance",
        type=float,
        default=0.12,
        help="Minimum ball-center clearance from every wall rectangle.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=10)
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
    return parser.parse_args(argv)


def validate_arguments(args):
    if not args.encoder_checkpoint.is_file():
        raise FileNotFoundError(
            f"Encoder checkpoint not found: {args.encoder_checkpoint}"
        )
    if not args.position_probe_checkpoint.is_file():
        raise FileNotFoundError(
            "Position-probe checkpoint not found: "
            f"{args.position_probe_checkpoint}"
        )
    if args.grid_resolution < 2:
        raise ValueError("grid-resolution must be at least 2")
    if args.wall_clearance < 0:
        raise ValueError("wall-clearance cannot be negative")
    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1")
    if args.log_every < 0:
        raise ValueError("log-every cannot be negative")


def maze_limits_from_walls(wall_rectangles):
    wall_rectangles = np.asarray(wall_rectangles, dtype=np.float32)
    if wall_rectangles.ndim != 2 or wall_rectangles.shape[1] != 4:
        raise ValueError("wall_rectangles must have shape [num_walls, 4]")
    if len(wall_rectangles) == 0:
        raise ValueError("at least one wall rectangle is required")

    x_min = wall_rectangles[:, 0].min()
    x_max = (wall_rectangles[:, 0] + wall_rectangles[:, 2]).max()
    y_min = wall_rectangles[:, 1].min()
    y_max = (wall_rectangles[:, 1] + wall_rectangles[:, 3]).max()
    return (float(x_min), float(x_max)), (float(y_min), float(y_max))


def generate_valid_grid_positions(
    x_limits,
    y_limits,
    wall_rectangles,
    grid_resolution,
    wall_clearance,
):
    if grid_resolution < 2:
        raise ValueError("grid_resolution must be at least 2")
    if wall_clearance < 0:
        raise ValueError("wall_clearance cannot be negative")

    wall_rectangles = np.asarray(wall_rectangles, dtype=np.float32)
    if wall_rectangles.ndim != 2 or wall_rectangles.shape[1] != 4:
        raise ValueError("wall_rectangles must have shape [num_walls, 4]")

    x_values = np.linspace(*x_limits, grid_resolution, dtype=np.float32)
    y_values = np.linspace(*y_limits, grid_resolution, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")
    positions = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)

    blocked = np.zeros(len(positions), dtype=bool)
    for x, y, width, height in wall_rectangles:
        blocked |= (
            (positions[:, 0] >= x - wall_clearance)
            & (positions[:, 0] <= x + width + wall_clearance)
            & (positions[:, 1] >= y - wall_clearance)
            & (positions[:, 1] <= y + height + wall_clearance)
        )

    valid_positions = positions[~blocked]
    if len(valid_positions) == 0:
        raise ValueError("the requested grid contains no collision-free points")
    return valid_positions


def decode_grid_positions(
    env,
    positions,
    encoder,
    position_probe,
    position_mean,
    position_std,
    device,
    batch_size,
    precision,
    log_every=0,
):
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape [num_positions, 2]")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")

    point_environment = env.unwrapped.point_env
    original_qpos = point_environment.data.qpos.copy()
    original_qvel = point_environment.data.qvel.copy()
    predictions = []
    num_batches = (len(positions) + batch_size - 1) // batch_size

    try:
        with torch.inference_mode():
            for batch_index, start in enumerate(
                range(0, len(positions), batch_size),
                start=1,
            ):
                batch_positions = positions[start:start + batch_size]
                frames = []
                for position in batch_positions:
                    qpos = original_qpos.copy()
                    qvel = np.zeros_like(original_qvel)
                    qpos[:2] = position
                    point_environment.set_state(qpos, qvel)
                    frames.append(validate_frame(env.render()).copy())

                frame_batch = torch.from_numpy(np.stack(frames))
                images = preprocess_frames(frame_batch).to(device=device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=precision == "bf16",
                ):
                    latents = encoder(images)
                    decoded = decode_positions(
                        position_probe=position_probe,
                        latents=latents,
                        position_mean=position_mean,
                        position_std=position_std,
                    )
                predictions.append(decoded.float().cpu())

                if log_every and (
                    batch_index % log_every == 0
                    or batch_index == num_batches
                ):
                    print(f"batch {batch_index}/{num_batches}")
    finally:
        point_environment.set_state(original_qpos, original_qvel)

    return torch.cat(predictions).numpy()


def save_grid_diagnostics(
    positions,
    predictions,
    wall_rectangles,
    output_dir,
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    positions = np.asarray(positions, dtype=np.float32)
    predictions = np.asarray(predictions, dtype=np.float32)
    wall_rectangles = np.asarray(wall_rectangles, dtype=np.float32)
    if predictions.shape != positions.shape:
        raise ValueError("predictions must have the same shape as positions")

    errors = np.linalg.norm(predictions - positions, axis=1)
    max_error_index = int(errors.argmax())
    summary = {
        "num_positions": int(len(positions)),
        "mean_error": float(errors.mean()),
        "median_error": float(np.median(errors)),
        "p90_error": float(np.percentile(errors, 90)),
        "max_error": float(errors[max_error_index]),
        "max_error_position": positions[max_error_index].tolist(),
        "prediction_at_max_error": predictions[max_error_index].tolist(),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "position_probe_grid_error.png"
    data_path = output_dir / "position_probe_grid_error.npz"
    summary_path = output_dir / "summary.json"

    np.savez_compressed(
        data_path,
        positions=positions,
        predictions=predictions,
        errors=errors,
        wall_rectangles=wall_rectangles,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(9, 7))
    for wall_index, (x, y, width, height) in enumerate(wall_rectangles):
        axis.add_patch(Rectangle(
            (x, y),
            width,
            height,
            facecolor="#b98247",
            edgecolor="#5f3d1f",
            linewidth=1.0,
            label="wall" if wall_index == 0 else None,
        ))

    error_map = axis.scatter(
        positions[:, 0],
        positions[:, 1],
        c=errors,
        cmap="inferno",
        marker="s",
        s=32,
        linewidths=0,
        zorder=3,
    )
    axis.scatter(
        *positions[max_error_index],
        marker="x",
        color="#00ffff",
        s=80,
        linewidth=2,
        zorder=4,
        label="maximum error",
    )
    colorbar = figure.colorbar(error_map, ax=axis, pad=0.02)
    colorbar.set_label("Euclidean decoding error")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("True maze x")
    axis.set_ylabel("True maze y")
    axis.set_title(
        "Position-probe error with the environment camera\n"
        f"mean={summary['mean_error']:.3f}, "
        f"p90={summary['p90_error']:.3f}, "
        f"max={summary['max_error']:.3f}"
    )
    axis.grid(alpha=0.15)
    axis.legend(loc="upper left", bbox_to_anchor=(1.18, 1.0))
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    paths = {
        "figure_path": str(figure_path),
        "data_path": str(data_path),
        "summary_path": str(summary_path),
    }
    return paths, summary


def main(argv=None):
    args = parse_args(argv)
    validate_arguments(args)
    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    encoder = load_frozen_ijepa_encoder(
        checkpoint_path=args.encoder_checkpoint,
        device=device,
    )
    position_probe, probe_metadata, probe_epoch = load_probe_checkpoint(
        args.position_probe_checkpoint,
        device,
    )
    position_mean = probe_metadata["position_mean"]
    position_std = probe_metadata["position_std"]

    environment = create_environment(
        environment_id=args.environment_id,
        max_micro_steps=1,
    )
    try:
        environment.reset(seed=0)
        hide_goal_marker(environment)
        wall_rectangles = extract_wall_rectangles(environment)
        x_limits, y_limits = maze_limits_from_walls(wall_rectangles)
        positions = generate_valid_grid_positions(
            x_limits=x_limits,
            y_limits=y_limits,
            wall_rectangles=wall_rectangles,
            grid_resolution=args.grid_resolution,
            wall_clearance=args.wall_clearance,
        )
        print(f"Device: {device}, precision: {precision}")
        if probe_epoch is not None:
            print(f"Position-probe epoch: {probe_epoch + 1}")
        print(f"Evaluating {len(positions)} collision-free positions")
        predictions = decode_grid_positions(
            env=environment,
            positions=positions,
            encoder=encoder,
            position_probe=position_probe,
            position_mean=position_mean,
            position_std=position_std,
            device=device,
            batch_size=args.batch_size,
            precision=precision,
            log_every=args.log_every,
        )
    finally:
        environment.close()

    paths, summary = save_grid_diagnostics(
        positions=positions,
        predictions=predictions,
        wall_rectangles=wall_rectangles,
        output_dir=args.output_dir,
    )
    print(
        "Errors: "
        f"mean={summary['mean_error']:.6f}, "
        f"median={summary['median_error']:.6f}, "
        f"p90={summary['p90_error']:.6f}, "
        f"max={summary['max_error']:.6f}"
    )
    print(f"Maximum-error position: {summary['max_error_position']}")
    print(f"Saved heatmap to: {paths['figure_path']}")
    print(f"Saved measurements to: {paths['data_path']}")
    print(f"Saved summary to: {paths['summary_path']}")


if __name__ == "__main__":
    main()

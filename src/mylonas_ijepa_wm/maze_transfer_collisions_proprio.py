"""Compare zero-shot collision predictions on the trained and rotated U mazes.

The starts and actions are identical between layouts. Only the rendered maze
changes for the world model, while the simulator supplies paired outcomes.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from src.mylonas_ijepa_wm import plan_pointmaze as base
from src.mylonas_ijepa_wm import plan_pointmaze_proprio as planner
from src.mylonas_ijepa_wm import wm, wm_proprio


MAZE_MAPS = {
    "original": [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ],
    "vertical": [
        [1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ],
}


@dataclass(frozen=True)
class CollisionCase:
    name: str
    start_cell: tuple[int, int]
    direction: tuple[float, float]
    changed_cell: tuple[int, int]


COLLISION_CASES = (
    CollisionCase("new_wall", (1, 1), (1.0, 0.0), (1, 2)),
    CollisionCase("removed_wall", (3, 1), (0.0, 1.0), (2, 1)),
)


def maze_map(layout):
    if layout not in MAZE_MAPS:
        raise ValueError(f"unknown maze layout: {layout}")
    return [row.copy() for row in MAZE_MAPS[layout]]


def cell_xy(cell):
    row, column = cell
    return np.asarray([column - 2.0, 2.0 - row], dtype=np.float32)


def trial_inputs(case, offset, force):
    if not 0 <= offset < 0.5:
        raise ValueError("offset must keep the initial ball inside its cell")
    if not 0 < force <= 1:
        raise ValueError("force must be in (0, 1]")
    direction = np.asarray(case.direction, dtype=np.float32)
    start_xy = cell_xy(case.start_cell) + offset * direction
    macro_action = np.tile(direction * force, 5).astype(np.float32)
    return start_xy, macro_action


def forward_progress(case, start_xy, end_xy):
    return float(np.dot(
        np.asarray(end_xy, dtype=np.float32) - np.asarray(start_xy, dtype=np.float32),
        np.asarray(case.direction, dtype=np.float32),
    ))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-checkpoint", type=Path, default=base.DEFAULT_ENCODER_CHECKPOINT)
    parser.add_argument("--world-model-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offsets", type=float, nargs="+", default=[0.25, 0.35])
    parser.add_argument("--forces", type=float, nargs="+", default=[0.7, 1.0])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    return parser.parse_args(argv)


def reset_to_state(env, start_xy, seed):
    env.reset(seed=seed)
    base.hide_goal_marker(env)
    env.unwrapped.point_env.set_state(
        np.asarray(start_xy, dtype=np.float64), np.zeros(2, dtype=np.float64)
    )
    point = env.unwrapped.point_env
    return np.concatenate((point.data.qpos[:2], point.data.qvel[:2])).astype(np.float32)


def simulate_macro(env, macro_action):
    states = []
    for action in macro_action.reshape(5, 2):
        observation, _, terminated, truncated, _ = env.step(action)
        states.append(np.asarray(observation["observation"], dtype=np.float32).copy())
        if terminated or truncated:
            break
    if len(states) != 5:
        raise RuntimeError("trial ended before its five environment actions")
    return np.asarray(states)


def predict_macro(model, latent, start_state, macro_action, metadata, device, precision):
    mean = torch.as_tensor(metadata["state_mean"], device=device, dtype=torch.float32)
    std = torch.as_tensor(metadata["state_std"], device=device, dtype=torch.float32)
    state = torch.as_tensor(start_state, device=device, dtype=torch.float32)
    history_length = model.num_hist
    visual_history = latent.unsqueeze(0).repeat(history_length, 1, 1).unsqueeze(0)
    state_history = ((state - mean) / std).repeat(history_length, 1).unsqueeze(0)
    action_history = torch.zeros(history_length - 1, 10, device=device)
    candidates = torch.as_tensor(macro_action, device=device).reshape(1, 1, 10)
    _, predicted = planner.rollout_action_sequences(
        model, visual_history, state_history, action_history, candidates,
        device=device, chunk_size=1,
        action_mean=metadata.get("action_mean"),
        action_std=metadata.get("action_std"), precision=precision,
    )
    return (predicted[0, 0].float() * std + mean).detach().cpu().numpy()


def save_plot(records, output_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, constrained_layout=True)
    colors = {"original": "#2563eb", "vertical": "#dc2626"}
    for ax, case in zip(axes, COLLISION_CASES):
        for layout in MAZE_MAPS:
            selected = [r for r in records if r["case"] == case.name and r["layout"] == layout]
            xs = np.arange(len(selected))
            actual = [r["actual_progress"] for r in selected]
            predicted = [r["predicted_progress"] for r in selected]
            ax.scatter(xs - 0.08, actual, marker="o", color=colors[layout],
                       label=f"{layout} actual")
            ax.scatter(xs + 0.08, predicted, marker="x", color=colors[layout],
                       label=f"{layout} predicted")
        ax.set_xticks(np.arange(len(selected)),
                      [f"{r['offset']:.2f}, {r['force']:.1f}" for r in selected], rotation=45)
        ax.set_title(case.name.replace("_", " "))
        ax.set_xlabel("start offset, action force")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("progress toward changed cell (maze units)")
    axes[1].legend(loc="best", fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize(records):
    groups = {}
    for layout in MAZE_MAPS:
        for case in COLLISION_CASES:
            selected = [r for r in records if r["layout"] == layout and r["case"] == case.name]
            groups[f"{layout}_{case.name}"] = {
                "mean_actual_progress": float(np.mean([r["actual_progress"] for r in selected])),
                "mean_predicted_progress": float(np.mean([r["predicted_progress"] for r in selected])),
                "mean_position_error": float(np.mean([r["position_error"] for r in selected])),
            }
    paired = {}
    for record in records:
        key = (record["case"], record["offset"], record["force"])
        paired.setdefault(key, {})[record["layout"]] = record
    return {
        "groups": groups,
        "mean_actual_layout_difference": float(np.mean([
            abs(pair["original"]["actual_progress"] - pair["vertical"]["actual_progress"])
            for pair in paired.values()
        ])),
        "mean_predicted_layout_difference": float(np.mean([
            abs(pair["original"]["predicted_progress"] - pair["vertical"]["predicted_progress"])
            for pair in paired.values()
        ])),
    }


def main(argv=None):
    args = parse_args(argv)
    for path in (args.encoder_checkpoint, args.world_model_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    for offset in args.offsets:
        trial_inputs(COLLISION_CASES[0], offset, 0.5)
    for force in args.forces:
        trial_inputs(COLLISION_CASES[0], 0.25, force)
    device = wm.resolve_device(args.device)
    precision = wm.resolve_precision(args.precision, device)
    model, metadata, epoch = wm_proprio.load_model_and_metadata(
        args.world_model_checkpoint, device
    )
    skip = int(metadata.get("physics_frame_skip", 1))
    if skip != 5:
        raise ValueError(f"expected a physics5 checkpoint, got physics_frame_skip={skip}")
    encoder = base.load_frozen_ijepa_encoder(args.encoder_checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for layout in MAZE_MAPS:
        env = base.create_environment("PointMaze_UMaze-v3", 10, camera="topdown",
                                      physics_frame_skip=skip, maze_map=maze_map(layout))
        try:
            for case in COLLISION_CASES:
                for offset in args.offsets:
                    start_xy, _ = trial_inputs(case, offset, args.forces[0])
                    start_state = reset_to_state(env, start_xy, seed=42)
                    start_frame = base.validate_frame(env.render()).copy()
                    latent = base.encode_frame(encoder, start_frame, device, precision)
                    frame_path = args.output_dir / f"{layout}_{case.name}_offset{offset:.2f}_start.png"
                    from PIL import Image
                    Image.fromarray(start_frame).save(frame_path)
                    for force in args.forces:
                        _, action = trial_inputs(case, offset, force)
                        actual_start = reset_to_state(env, start_xy, seed=42)
                        if not np.allclose(actual_start, start_state, atol=1e-5):
                            raise RuntimeError("paired trial did not reset to the same state")
                        predicted_state = predict_macro(
                            model, latent, start_state, action, metadata, device, precision
                        )
                        actual_states = simulate_macro(env, action)
                        actual_state = actual_states[-1]
                        label = f"{layout}_{case.name}_offset{offset:.2f}_force{force:.1f}"
                        Image.fromarray(base.validate_frame(env.render())).save(
                            args.output_dir / f"{label}_end.png"
                        )
                        record = {
                            "layout": layout, "case": case.name, "offset": offset,
                            "force": force, "start_state": start_state.tolist(),
                            "macro_action": action.tolist(),
                            "actual_states": actual_states.tolist(),
                            "actual_state": actual_state.tolist(),
                            "predicted_state": predicted_state.tolist(),
                            "actual_progress": forward_progress(case, start_xy, actual_state[:2]),
                            "predicted_progress": forward_progress(case, start_xy, predicted_state[:2]),
                            "position_error": float(np.linalg.norm(predicted_state[:2] - actual_state[:2])),
                        }
                        records.append(record)
                        print(label, "actual progress", f"{record['actual_progress']:.3f}",
                              "predicted progress", f"{record['predicted_progress']:.3f}")
        finally:
            env.close()
    payload = {
        "checkpoint": str(args.world_model_checkpoint), "checkpoint_epoch": epoch + 1,
        "physics_frame_skip": skip, "history": "initial frame and state repeated",
        "summary": summarize(records),
        "records": records,
    }
    (args.output_dir / "results.json").write_text(json.dumps(payload, indent=2))
    save_plot(records, args.output_dir / "progress_comparison.png")
    print("Saved", args.output_dir)


if __name__ == "__main__":
    main()

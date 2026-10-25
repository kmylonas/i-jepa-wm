"""Compare DotWall predictions at identical states/actions in two visual layouts."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.mylonas_ijepa_wm import plan_pointmaze as visual
from src.mylonas_ijepa_wm import wm, wm_proprio
from src.mylonas_ijepa_wm.environment_adapters import create_adapter
from src.mylonas_ijepa_wm.environment_configs import load_environment_config
from src.mylonas_ijepa_wm.wall_layout_manifest import (
    TRAIN_LAYOUTS, load_manifest, validate_checkpoint_manifest,
)


def build_matched_pairs() -> list[dict]:
    """Predeclared contrasting cases; their geometry does not depend on model output."""
    return [
        dict(name="door_y22_left_to_right", layouts=[[26, 22], [26, 38]],
             start=[21.0, 22.0], actions=[[1.0, 0.0]] * 5,
             reference_x=26, expected_crossed=[True, False]),
        dict(name="door_y38_right_to_left", layouts=[[26, 22], [26, 38]],
             start=[31.0, 38.0], actions=[[-1.0, 0.0]] * 5,
             reference_x=26, expected_crossed=[False, True]),
        dict(name="wall_x26_right_to_left", layouts=[[26, 22], [38, 22]],
             start=[31.0, 32.0], actions=[[-1.0, 0.0]] * 5,
             reference_x=26, expected_crossed=[False, True]),
        dict(name="wall_x38_left_to_right", layouts=[[26, 22], [38, 22]],
             start=[33.0, 32.0], actions=[[1.0, 0.0]] * 5,
             reference_x=38, expected_crossed=[True, False]),
    ]


def _crossed(start_x: float, end_x: float, reference_x: float) -> bool:
    return bool((start_x < reference_x < end_x) or (end_x < reference_x < start_x))


def _evaluate_case(adapter, encoder, model, metadata, pair, layout, seed, device, precision):
    adapter.reset(seed, layout=layout)
    adapter.env.set_init_state(np.asarray(pair["start"], dtype=np.float32))
    adapter.env.reset()
    start = adapter.state()
    if not np.allclose(start, pair["start"], atol=1e-5):
        raise ValueError(f"Simulator did not use requested start in {pair['name']}")
    initial_frame = adapter.render()
    latent = visual.encode_frame(encoder, initial_frame, device, precision)
    state_mean = metadata["state_mean"].to(device=device, dtype=torch.float32)
    state_std = metadata["state_std"].to(device=device, dtype=torch.float32)
    action = torch.as_tensor(
        np.asarray(pair["actions"], dtype=np.float32).reshape(1, 1, -1),
        device=device,
    )
    if metadata.get("normalize_actions", False):
        action = (action - metadata["action_mean"].to(device)) / metadata["action_std"].to(device)
    state = torch.as_tensor(start, device=device, dtype=torch.float32)
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=precision == "bf16",
        ):
            predicted_tokens, predicted_states = model(
                latent[None, None], ((state - state_mean) / state_std)[None, None], action,
            )
    predicted = (predicted_states[0, -1].float() * state_std + state_mean).cpu().numpy()

    positions = [start.tolist()]
    final_frame = initial_frame
    for primitive in pair["actions"]:
        outcome = adapter.step(primitive)
        positions.append(np.asarray(outcome.state).tolist())
        final_frame = outcome.frame
    actual = np.asarray(positions[-1], dtype=np.float32)
    actual_tokens = visual.encode_frame(encoder, final_frame, device, precision)
    token_mse = torch.mean((predicted_tokens[0, -1].float() - actual_tokens.float()).square())
    reference_x = pair["reference_x"]
    return dict(
        layout=list(layout), start=start.tolist(), actions=pair["actions"],
        actual_positions=positions, actual_final_position=actual.tolist(),
        predicted_final_position=predicted.tolist(),
        actual_crossed=_crossed(start[0], actual[0], reference_x),
        predicted_crossed=_crossed(start[0], predicted[0], reference_x),
        position_error=float(np.linalg.norm(predicted - actual)),
        token_mse=float(token_mse.item()),
    )


def evaluate_pair(adapter, encoder, model, metadata, pair, seed, device, precision) -> dict:
    cases = [
        _evaluate_case(adapter, encoder, model, metadata, pair, layout, seed, device, precision)
        for layout in pair["layouts"]
    ]
    if any(case["start"] != pair["start"] or case["actions"] != pair["actions"] for case in cases):
        raise ValueError(f"Matched inputs changed in {pair['name']}")
    actual = [case["actual_crossed"] for case in cases]
    if actual != pair["expected_crossed"] or actual[0] == actual[1]:
        raise ValueError(f"Simulator outcomes do not match predeclared contrast in {pair['name']}: {actual}")
    predicted = [case["predicted_crossed"] for case in cases]
    return dict(
        name=pair["name"], seed=seed, reference_x=pair["reference_x"],
        start=pair["start"], actions=pair["actions"],
        expected_crossed=pair["expected_crossed"], cases=cases,
        predicted_contrast=predicted[0] != predicted[1],
        both_correct=predicted == actual,
    )


def build_pair_figure(result: dict):
    """Show true primitive paths and one-step predicted endpoints for a matched pair."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    x_values = []
    y_values = []
    for case in result["cases"]:
        for position in [*case["actual_positions"], case["predicted_final_position"]]:
            x_values.append(position[0])
            y_values.append(position[1])
    # DotWall renders pixels 0..64. Expand only if a prediction leaves the frame.
    x_limits = (min(-0.5, min(x_values) - 2), max(64.5, max(x_values) + 2))
    y_limits = (max(64.5, max(y_values) + 2), min(-0.5, min(y_values) - 2))
    train_doors = sorted({door_y for _, door_y in TRAIN_LAYOUTS})
    rail_offsets = {
        door_y: (index - (len(train_doors) - 1) / 2) * 0.9
        for index, door_y in enumerate(train_doors)
    }
    for ax, case in zip(axes, result["cases"]):
        wall_x, door_y = case["layout"]
        for train_x, train_door_y in TRAIN_LAYOUTS:
            rail_x = train_x + rail_offsets[train_door_y]
            for bottom, top in ((0, train_door_y - 4), (train_door_y + 4, 64)):
                ax.plot([rail_x, rail_x], [bottom, top], color="#859443",
                        linestyle="--", linewidth=1.4, alpha=0.3, zorder=3,
                        label="Training layout (faint)")
        for bottom, top in ((0, door_y - 4), (door_y + 4, 64)):
            ax.add_patch(Rectangle(
                (wall_x - 3, bottom), 6, top - bottom,
                facecolor="#29343d", edgecolor="none", zorder=2,
                label="Current wall" if bottom == 0 else None,
            ))
        ax.axvline(result["reference_x"], color="#8b4aa8", linestyle=":",
                   linewidth=1.8, zorder=3,
                   label=f"Virtual test line x={result['reference_x']}")
        positions = np.asarray(case["actual_positions"], dtype=float)
        ax.plot(positions[:, 0], positions[:, 1], color="#1678a7", linewidth=2.5,
                marker="o", markersize=3.5, zorder=4, label="Actual trajectory")
        ax.scatter(*positions[0], s=90, color="#101820", edgecolor="white",
                   linewidth=1, zorder=5, label="Shared start")
        ax.scatter(*case["predicted_final_position"], s=130, marker="X",
                   color="#d4542d", edgecolor="white", linewidth=0.8,
                   zorder=6, label="Predicted endpoint")
        actual = "yes" if case["actual_crossed"] else "no"
        predicted = "yes" if case["predicted_crossed"] else "no"
        ax.set_title(f"Wall x={wall_x} · Door y={door_y}\n"
                     f"Test x={result['reference_x']}: real {actual} · WM {predicted}")
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (environment units)")
        ax.grid(alpha=0.13, zorder=0)
    axes[0].set_ylabel("y (environment units)")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=6,
               frameon=False, bbox_to_anchor=(0.5, 0.025))
    fig.suptitle(result["name"].replace("_", " "))
    fig.text(0.5, 0.145,
             "Faint rails show nine separate training layouts; rails at each x are offset for visibility.",
             ha="center", fontsize=9)
    fig.text(0.5, 0.12,
             "Purple test line is a scoring threshold, not a wall. The WM predicts only the endpoint after five actions.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.2, 1, 0.93))
    return fig


def save_pair_figure(result: dict, output: Path) -> Path:
    from matplotlib import pyplot as plt

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = build_pair_figure(result)
    try:
        figure.savefig(output, dpi=170)
    finally:
        plt.close(figure)
    return output


def evaluate_pairs(
    checkpoint: Path, encoder_checkpoint: Path, manifest: dict,
    output: Path, env_config: Path, device_name: str = "auto",
    precision_name: str = "auto", seed: int = 10000,
) -> dict:
    config = load_environment_config("wall", env_config)
    device = wm.resolve_device(device_name)
    precision = wm.resolve_precision(precision_name, device)
    model, metadata, epoch = wm_proprio.load_model_and_metadata(checkpoint, device)
    validate_checkpoint_manifest(metadata, manifest)
    if metadata.get("environment") != "wall" or model.num_hist != 1:
        raise ValueError("Visual pairs require a one-history Wall checkpoint")
    pairs = build_matched_pairs()
    test_layouts = {tuple(layout) for layout in manifest["test_layouts"]}
    if any(tuple(layout) not in test_layouts for pair in pairs for layout in pair["layouts"]):
        raise ValueError("A visual pair layout is outside the held-out test split")
    if config.actions_per_model_step != 5:
        raise ValueError("Visual pairs require five actions per model step")
    encoder = visual.load_frozen_ijepa_encoder(encoder_checkpoint, device)
    adapter = create_adapter(config, max_micro_steps=5)
    try:
        results = [
            evaluate_pair(adapter, encoder, model, metadata, pair, seed, device, precision)
            for pair in pairs
        ]
    finally:
        adapter.close()
    cases = [case for result in results for case in result["cases"]]
    payload = dict(
        checkpoint=str(checkpoint), checkpoint_epoch=epoch,
        encoder_checkpoint=str(encoder_checkpoint), seed=seed,
        summary=dict(
            pairs=len(results), cases=len(cases),
            predicted_contrasts=sum(result["predicted_contrast"] for result in results),
            both_correct_pairs=sum(result["both_correct"] for result in results),
            crossing_matches=sum(case["predicted_crossed"] == case["actual_crossed"] for case in cases),
            mean_position_error=float(np.mean([case["position_error"] for case in cases])),
        ),
        pairs=results,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure_dir = output.with_name(f"{output.stem}_figures")
    for result in results:
        result["figure_path"] = str(save_pair_figure(result, figure_dir / f"{result['name']}.png"))
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    args = parser.parse_args(argv)
    result = evaluate_pairs(
        args.checkpoint, args.encoder_checkpoint, load_manifest(args.episode_manifest),
        args.output, args.env_config, args.device, args.precision, args.seed,
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved visual pairs to {args.output}")


if __name__ == "__main__":
    main()
